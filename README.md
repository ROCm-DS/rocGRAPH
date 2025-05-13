# rocGRAPH

rocGraph is a collection of algorithm implementations focused on GPU-accelerated graph analytics, it supports the creation and manipulation of graphs followed by the execution of scalable fast graph algorithms. It is implemented on top of AMD
[ROCm](https://github.com/ROCm/ROCm) runtime and toolchains. rocGRAPH is created using the [HIP](https://github.com/ROCm/HIP/) programming
language and optimized for AMD's latest discrete GPUs.

This is an initial, source-only release.

## Documentation

Documentation for rocGRAPH is in progress.

## Requirements

* Git
* CMake (3.5 or later)
* AMD [ROCm](https://github.com/ROCm/ROCm) 6.4.0 platform or later

rocGraph builds against the following header implementation (fetched by Cmake during the build):
* [fmt](https://github.com/fmtlib/fmt)
* [spdlog](https://github.com/gabime/spdlog)
* [hipCollections](https://github.com/ROCm/hipCollections)
* [rmm-rocm](https://github.com/ROCm/rmm-rocm)
* [libhipcxx](https://github.com/ROCm/libhipcxx)
* [raft](https://github.com/ROCm/raft)

It links to the following librarieas:
* [hipcub](https://github.com/ROCm/hipcub)
* [rocthrust](https://github.com/ROCm/rocthrust)
* [hiprand](https://github.com/ROCm/hiprand)

For building the tests:
* [GoogleTest](https://github.com/google/googletest) (fetched by Cmake during the build)

## Build and install

To build rocGRAPH, you can use our bash helper script (for Ubuntu, Centos, RHEL, Fedora, SLES, openSUSE-Leap) or you can
perform a manual build (for all supported platforms).

* Bash helper script (`install.sh`):
  This script, which is located in the root of this repository, builds and installs rocGRAPH on Ubuntu
  with a single command. Note that this option doesn't allow much customization and hard-codes
  configurations that can be specified through invoking CMake directly. Some commands in the script
  require sudo access, so it may prompt you for a password.

  ```bash
  git clone https://github.com/ROCm/rocGRAPH.git # Clone rocGRAPH using git
  ./install.sh -h  # shows help
  ./install.sh -id # builds library, dependencies, then installs (the `-d` flag only needs to be passed once on a system)
  ./install.sh -ic # builds library and clients for testing.
  ```

* Manual build:
  If you use a distribution other than Ubuntu, or would like more control over the build process,
  run cmake manually.

  For instance to build the `Release` build type, on 32 cores, for only the `gfx90a` architecture:

  ```bash
  git clone https://github.com/ROCm/rocGRAPH.git # Clone rocGRAPH using git
  cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=toolchain-linux.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx90a"
  nice cmake --build build --parallel 32
  build/release/clients/staging/rocgraph-test
  ```
  To run the tests execute in the root directory:
  ```bash
  build/clients/staging/rocgraph-test
  ```

## Issues

To submit an issue, a bug, or a feature request, use the GitHub
[issue tracker](https://github.com/ROCm/rocGRAPH/issues).

## License

Our [license file](https://github.com/ROCm/rocGRAPH) is located in the main
repository.
