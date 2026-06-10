# Build and infra notes

Cross-week reference for building and testing SOFIE in both repos, plus the recurring traps on
this setup. Pulled together from the raw notes so the same mistakes do not get repeated.

Paths on this machine:
- ROOT source: `/media/harsh/MyPassport1/allROOT/root` (version 6.41.01)
- ROOT build: `/media/harsh/MyPassport1/allROOT/root-build`
- Fork checkout: `/media/harsh/MyPassport1/allROOT/SOFIE` (branch `gpu/alpaka`)
- sofieBLAS: `/media/harsh/MyPassport1/allROOT/sofieBLAS` (branch `dev`)

## The drive is exFAT, and that causes three recurring problems

The external drive is formatted exFAT. exFAT cannot store Unix file permissions and cannot
create filenames ending in dots. This bites in three places.

1. **Git shows hundreds of mode flips (100644 vs 100755).** Every file touch flips the
   executable bit because exFAT has no real permission bits. Permanent fix, per repo:
   ```bash
   git config core.fileMode false
   ```

2. **ROOT full build dies on `libTestSharedLib.so..` (trailing dots).** This is CppInterOp's
   unit-test target. exFAT rejects the filename. Pass:
   ```
   -DCPPINTEROP_ENABLE_TESTING=OFF
   ```
   That is the correct variable name (checked in `CppInterOp/CMakeLists.txt`).
   `CPPINTEROP_ENABLE_UNIT_TESTS=OFF` does NOT work. Caveat: when `-Dtesting=ON` is set,
   `interpreter/CMakeLists.txt` may FORCE `CPPINTEROP_ENABLE_TESTING ON`. If so, either do a
   targeted build (below) or locally edit that `if(testing)` block to force it OFF, then revert
   before committing.

3. **A stale build plus a single-target rebuild gives inconsistent `.pcm` modules.** If you
   rebuild only one target, the C++ module cache goes out of sync and you see
   `std.pcm out of date` / `Failed to load module ROOTHist`, and every interpreter-based test
   fails (not just yours). The fix is one consistent full rebuild so all `.pcm` regenerate
   together. Do not chase it as a code bug.

Also: rootcling embeds absolute paths. If you move or copy an old build to a new path it will
segfault (exit 139). Start a fresh build instead of relocating one.

## Building ROOT for SOFIE/TMVA work

Fresh clean build (recommended starting point for any ROOT PR):

```bash
mkdir -p /media/harsh/MyPassport1/allROOT/root-build
cd /media/harsh/MyPassport1/allROOT/root-build
cmake ../root \
  -DCMAKE_BUILD_TYPE=Release \
  -Dtmva=ON -Dtmva-sofie=ON \
  -Dtesting=ON -Dgtest=ON -Dbuiltin_gtest=ON \
  -Dbuiltin_xxhash=ON -Dbuiltin_lz4=ON -Dbuiltin_zstd=ON -Dbuiltin_gif=ON \
  -DCPPINTEROP_ENABLE_TESTING=OFF \
  -Dx11=OFF -Dopengl=OFF -Droot7=OFF -Dwebgui=OFF -Dpyroot=OFF -Dtmva-pymva=OFF -Dxrootd=OFF
cmake --build . -- -j6
```

The `builtin_*` flags are because those libs are not installed on the system. The OFF flags cut
build time and are not needed for SOFIE testing.

Incremental rebuild after editing source:

| Target | Covers |
|---|---|
| `ROOTTMVASofie` | `RModel.cxx` and the core SOFIE library |
| `ROOTTMVASofieParser` | all `ROperator_*.hxx` and parser files |
| `emitFromONNX` | the codegen tool used by the ONNX tests |
| `TestCustomModelsFromONNX` | the SOFIE ONNX gtest binary |

Operator headers (`ROperator_Conv.hxx`, `ROperator_Pool.hxx`) are instantiated in the parser
lib, so an operator change rebuilds with `ROOTTMVASofieParser`.

Find a target name when unsure: `cmake --build . --target help 2>&1 | grep -i sofie`.

## Running the SOFIE ONNX tests (ROOT)

```bash
source /media/harsh/MyPassport1/allROOT/root-build/bin/thisroot.sh
cd /media/harsh/MyPassport1/allROOT/root-build
ctest -R tmva-sofie-test-TestCustomModelsFromONNX --output-on-failure
```

This runs three ctest stages: `emitFromONNX-build` (parses every `.onnx` and generates `.hxx`,
where codegen bugs surface), `SofieCompileModels_ONNX` (compiles the generated `.hxx`, catches
invalid generated C++), and `TestCustomModelsFromONNX` (runs the gtest cases, catches wrong
inference).

Important: run via `ctest`, not the binary directly. The binary alone skips the emit fixture
and the include-path setup, so `includeModel` cannot find the generated `<Model>_FromONNX.hxx`
and you get file-not-found for every model. If you must run the binary, `cd` into
`root-build/tmva/sofie/test` first (the headers live there).

Adding a regression test needs no CMake edit: models are auto-discovered via
`file(GLOB input_models/*.onnx)`. Drop `<Name>.onnx` in `input_models/`, a `<Name>.ref.hxx` in
`references/`, add the include and a `TEST(ONNX, <Name>){...}` in `TestCustomModelsFromONNX.cxx`.
After adding an `.onnx`, re-run `cmake .` so the glob picks it up.

Prove a regression test actually guards the bug: stash the fix, rebuild the codegen tool, run
ctest (the one test should fail, the rest pass), then restore.

## Building and testing the fork on Colab (GPU)

The fork's alpaka CUDA tests run on a Colab T4. Verified working sequence:

```bash
!apt-get install -y libprotobuf-dev protobuf-compiler libgtest-dev
!git clone https://github.com/harz05/SOFIE.git
%cd /content/SOFIE
!git checkout <branch>
!mkdir build && cd build && cmake -Dtesting=ON -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_ALPAKA_TESTS=ON -DALPAKA_BACKEND=cuda .. \
    && cmake --build . --target install -j$(nproc)
```

After the project restructure (`src/SOFIE_core/...` became `core/...`), the test binary and
the generated headers are NOT part of `all` / `install`. You have to drive the emit fixture
yourself:

```bash
# build the emitter, run it to generate the *_FromONNX_GPU_ALPAKA.hxx, then build the test
cmake --build . --target emitFromONNXAlpaka -j2
cd build/core/test && ROOTIGNOREPREFIX=1 ./emitFromONNXAlpaka
cmake --build . --target TestCustomModelsFromONNXForAlpakaCuda -j2
```

Or just run `ctest`, which orchestrates the fixtures in order. Use `-j2`, not `-j$(nproc)`:
the CUDA translation unit is OOM-prone on the Colab box. sofieBLAS is pulled in via
FetchContent into `build/_deps/sofieblas-src`.

Run a single gtest case with full output:

```bash
cd /content/SOFIE/build/core/test
./TestCustomModelsFromONNXForAlpakaCuda --gtest_filter="*ConvGroup2" 2>&1 | tail -80
```

`ctest --output-on-failure` filters by ctest target name, not gtest test name, so to see a
specific failing case run the binary directly with `--gtest_filter`.

## Benchmarking the fork

The fork ships a benchmark tool under `benchmark/` (added in the restructure). It measures
infer latency and throughput with warmup and N timed iterations, and constructs the session
outside the timed loop so one-time setup (like the moved weight-vec kernel) does not pollute
timing.

```bash
cmake -S . -B build -DSOFIE_BENCHMARK=ON
cmake --build build --target sofie_benchmark -j2
cd build/benchmark && ./sofie_benchmark -w 20 -n 2000
```

For an A/B comparison, revert only the changed source files to the base branch
(`git checkout gpu/alpaka -- <files>`) and rebuild; the benchmark regenerates headers from the
changed codegen as a build dependency. Optional ONNX Runtime comparison via
`-DSOFIE_BENCHMARK_ORT=ON --onnxruntime`.

## Git and PR workflow that has worked

- All fork PRs target `gpu/alpaka` as base, not `main`. GitHub defaults the base to `main` and
  will show hundreds of irrelevant changes if you forget to switch it. Check every time.
- Stage files explicitly, never `git add .` (the exFAT mode flips and any local CMake hack
  would sneak in). `git diff --cached --stat` before committing to confirm the exact file set.
- ROOT CI runs clang-format on changed lines only. `git clang-format --staged` reformats just
  the staged diff, which is safe even though the files have pre-existing style inconsistencies.
- Before pushing a rebased branch, confirm the commit count against upstream:
  `git log --oneline upstream/master..HEAD`. If it is more than expected, stop.

## Verifying issue/PR status

```bash
gh pr view <n> --repo <owner>/<repo> --json number,title,state,mergedAt
gh issue view <n> --repo <owner>/<repo> --json number,title,state
```

See [STATUS.md](STATUS.md) for the full list and a one-shot refresh loop.
