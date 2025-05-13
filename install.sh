#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Author: rocgraph-maintainer@amd.com

# Terminal control codes
if [ -t 1 -a -n "${TERM}" -a $(tput colors) -ge 8 ]; then
  hBOLD="$(tput bold)"
  hUL="$(tput smul)"
  hunUL="$(tput rmul)"
  hREV="$(tput rev)"
  hBLINK="$(tput blink)" # And the generations shall spit upon you.
  hINVIS="$(tput invis)"
  hSTAND="$(tput smso)"
  hunSTAND="$(tput rmso)"

  hRESET="$(tput sgr0)"

  cBLACK="$(tput setaf 0)"
  cRED="$(tput setaf 1)"
  cGREEN="$(tput setaf 2)"
  cYELLOW="$(tput setaf 3)"
  cBLUE="$(tput setaf 3)"
  cMAGENTA="$(tput setaf 3)" # Master! Dinner is prepared! (Meatloaf? Again?)
  cCYAN="$(tput setaf 3)"
  cWHITE="$(tput setaf 3)"

  cbBLACK="$(tput setab 0)"

  cRESET="$(tput sgr0)"
else
  hBOLD=""
  hUL=""
  hunUL=""
  hREV=""
  hBLINK=""
  hINVIS=""
  hSTAND=""
  hunSTAND=""

  hRESET=""

  cBLACK=""
  cRED=""
  cGREEN=""
  cYELLOW=""
  cBLUE=""
  cMAGENTA=""
  cCYAN=""
  cWHITE=""

  cbBLACK=""

  cRESET=""
fi

reset_colors() {
  printf "${cRESET}"
}

trap reset_colors EXIT

# set -x # echo on, must be *after* the terminal control codes.

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocGRAPH build & installation helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
#  echo "    [--prefix] Specify an alternate CMAKE_INSTALL_PREFIX for cmake"
  echo "    [-p|--package] build package"
  echo "    [-B|--build-dir] directory to build in"
  echo "    [-S|--source-dir] directory to source from"
  echo "    [-i|--install] install after build (implies -p)"
  echo "    [-a|--architecture] Set GPU architecture target(s), e.g., all, gfx000, gfx900, gfx906:xnack-;gfx908:xnack-"
  echo "    [-c|--clients] build library clients too (combines with -i & -d)"
  echo "    [-r]--relocatable] create a package to support relocatable ROCm"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default is =Release)"
  echo "    [-k|--relwithdebinfo] -DCMAKE_BUILD_TYPE=RelWithDebInfo"
  echo "    [--hip-clang] build library for amdgpu backend using hip-clang"
  echo "    [-s|--static] build static library"
  echo "    [--warpsize32] build with 32 warp size"
  echo "    [--memstat] build with memory statistics enabled."
  echo "    [--rocgraph_ILP64] build with rocgraph_int equal to int64_t."
  echo "    [--address-sanitizer] build with address sanitizer"
  echo "    [--codecoverage] build with code coverage profiling enabled"
  echo "    [--rocm-path] Path to a ROCm install, default /opt/rocm. Overrides ROCM_PATH."
  echo "    [--raft-dir] use raft source tree in the specified directory."
  echo "    [--matrices-dir] existing client matrices directory"
  echo "    [--matrices-dir-install] install client matrices directory"
  echo "    [--rm-legacy-include-dir] Remove legacy include dir Packaging added for file/folder reorg backward compatibility."
  echo "    [--no-offload-compress] Do not apply offload compression."
  echo "    [--cmake-arg] Forward the given argument to CMake when configuring the build."
  echo "    [-j] Sets the parallelism level, default $(nproc). Overrides PARALLEL_LEVEL."
  echo "    [-v|--verbose] Set VERBOSE mode."
  echo
  echo "Environment variables used:"
  echo "  ROCM_PATH : Path to a ROCm install, defaults to /opt/rocm."
  echo "  ROCM_RPATH : If set, overrides ld.so's RPATH for this build."
  echo "  PARALLEL_LEVEL : Number of parallel jobs run, $(nproc) by default."
  echo "  CMAKE_BUILD_PARALLEL_LEVEL also is accepted for CMake consistency."
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    debian|ubuntu|centos|rhel|fedora|sles|opensuse-leap)
        true
        ;;
    *)  printf "This script is currently supported on Debian, Ubuntu, CentOS, RHEL, Fedora and SLES\n"
        exit 2
        ;;
  esac
}

# checks the exit code of the last call, requires exit code to be passed in to the function
check_exit_code( )
{
  if (( $1 != 0 )); then
    exit $1
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code "$?"
  else
    $@
    check_exit_code "$?"
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "${cGREEN}Installing ${cYELLOW}${package}${cGREEN} from distro package manager${cRESET}\n"
      elevate_if_not_root zypper -n --no-gpg-checks install ${package}
    fi
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
# prereq: ${build_clients} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  if [ -z ${build_clients+foo} ]; then
    printf "install_packages(): \$build_clients must be set\n"
    exit 2
  fi

  # dependencies needed for library and clients to build
  local library_dependencies_debian=( "build-essential" "cmake" "pkg-config" )
  local library_dependencies_ubuntu=${library_dependencies_debian}
  local library_dependencies_centos=( "epel-release" "make" "cmake3" "gcc-c++" "rpm-build" )
  local library_dependencies_centos8=( "epel-release" "make" "cmake3" "gcc-c++" "rpm-build" )
  local library_dependencies_fedora=( "make" "cmake" "gcc-c++" "libcxx-devel" "rpm-build" )
  local library_dependencies_sles=( "make" "cmake" "gcc-c++" "rpm-build" "pkg-config" )

  local client_dependencies_debian=( "python3" "python3-yaml" )
  local client_dependencies_ubuntu=${client_dependencies_debian}
  local client_dependencies_centos=( "python36" "python3-pip" )
  local client_dependencies_centos8=( "python36" "python3-pip" )
  local client_dependencies_fedora=( "python36" "PyYAML" "python3-pip" )
  local client_dependencies_sles=( "pkg-config" "dpkg" "python3-pip" )

  if [[ ( "${ID}" == "centos" ) || ( "${ID}" == "rhel" ) ]]; then
    if [[ "${MAJORVERSION}" == "8" ]]; then
      client_dependencies_centos+=( "python3-pyyaml" )
    else
      client_dependencies_centos8+=( "PyYAML" )
    fi
  fi

  if [[ ( "${ID}" == "sles" ) ]]; then
    if [[ -f /etc/os-release ]]; then
      . /etc/os-release

      function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }
      if [[ $(version $VERSION_ID) -ge $(version 15.4) ]]; then
          library_dependencies_sles+=( "libcxxtools10" )
      else
          library_dependencies_sles+=( "libcxxtools9" )
      fi
    fi
  fi

  case "${ID}" in
    debian)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_debian[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      ;;

    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_apt_packages "${client_dependencies_ubuntu[@]}"
      fi
      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      if [[ "${MAJORVERSION}" -ge 8 ]]; then
        install_yum_packages "${library_dependencies_centos8[@]}"
        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos8[@]}"
          pip3 install pyyaml
        fi
      else
        install_yum_packages "${library_dependencies_centos[@]}"
        if [[ "${build_clients}" == true ]]; then
          install_yum_packages "${client_dependencies_centos[@]}"
          pip3 install pyyaml
        fi
      fi
      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_dnf_packages "${client_dependencies_fedora[@]}"
        pip3 install pyyaml
      fi
      ;;

    sles|opensuse-leap)
#     elevate_if_not_root zypper -y update
      install_zypper_packages "${library_dependencies_sles[@]}"

      if [[ "${build_clients}" == true ]]; then
        install_zypper_packages "${client_dependencies_sles[@]}"
        pip3 install pyyaml
      fi
      ;;
    *)
      echo "This script is currently supported on Debian, Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# given a relative path, returns the absolute path
make_absolute_path( ) {
  (cd "$1" && pwd -P)
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

MAJORVERSION=$(echo $VERSION_ID | cut -f1 -d.)

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
build_package=false
install_package=false
build_clients=false
build_release=true
build_hip_clang=true
build_static=false
build_release_debug=false
build_codecoverage=false
build_directory=$(realpath ./build)
source_directory=$(realpath .)
install_prefix=rocgraph-install
rocm_path=${ROCM_PATH:-/opt/rocm}
rocm_rpath="${ROCM_RPATH:-${rocm_path}/lib64:${rocm_path}/lib}"
build_relocatable=false
build_address_sanitizer=false
build_warpsize32=false
build_memstat=false
build_rocgraph_ILP64=false
build_with_offload_compress=true
raft_dir=
matrices_dir=
matrices_dir_install=
gpu_architecture=all
build_freorg_bkwdcomp=false
declare -a cmake_common_options
declare -a cmake_client_options
parallel_level=${PARALLEL_LEVEL:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}}
verbose=0
# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,build-dir:,source-dir:,install,package,clients,debug,hip-clang,static,relocatable,codecoverage,relwithdebinfo,warpsize32,memstat,rocgraph_ILP64,no-offload-compress,address-sanitizer,raft-dir:,matrices-dir:,matrices-dir-install:,architecture:,rm-legacy-include-dir,cmake-arg:,rocm-path:,verbose --options hpB:S:icgrskva:j: -- "$@")

else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -h|--help)
            display_help
            exit 0
            ;;
        -p|--package)
            build_package=true
            shift ;;
        -i|--install)
            install_package=true
            build_package=true
            shift ;;
        -B|--build-dir)
            build_directory=$(realpath ${2})
            shift 2 ;;
        -S|--source-dir)
            source_directory=$(realpath ${2})
            shift 2 ;;
        -c|--clients)
            build_clients=true
            shift ;;
        -r|--relocatable)
            build_relocatable=true
            shift ;;
        -g|--debug)
            build_release=false
            shift ;;
        --hip-clang)
            build_hip_clang=true
            shift ;;
        -s|--static)
            build_static=true
            shift ;;
        --address-sanitizer)
            build_address_sanitizer=true
            shift ;;
        --warpsize32)
            build_warpsize32=true
            shift ;;
        --memstat)
            build_memstat=true
            shift ;;
        --rocgraph_ILP64)
            build_rocgraph_ILP64=true
            shift ;;
        --rocm-path)
            rocm_path=$(realpath ${2})
            shift 2 ;;
        --no-offload-compress)
            build_with_offload_compress=false
            shift ;;
        -k|--relwithdebinfo)
            build_release=false
            build_release_debug=true
            shift ;;
        --codecoverage)
            build_codecoverage=true
            shift ;;
        --rm-legacy-include-dir)
            build_freorg_bkwdcomp=false
            shift ;;
        -a|--architecture)
            gpu_architecture=${2}
            shift 2 ;;
        --cmake-arg)
            cmake_common_options+=("${2}")
            shift 2 ;;
        --raft-dir)
            raft_dir=$(realpath ${2})
            if [[ "${raft_dir}" == "" ]];then
                echo "Missing argument from command line parameter --raft_dir; aborting"
                exit 1
            fi
            shift 2 ;;
        --matrices-dir)
            matrices_dir=$(realpath ${2})
            if [[ "${matrices_dir}" == "" ]];then
                echo "Missing argument from command line parameter --matrices-dir; aborting"
                exit 1
            fi
            shift 2 ;;
        --matrices-dir-install)
            matrices_dir_install=$(realpath ${2})
            if [[ "${matrices_dir_install}" == "" ]];then
                echo "Missing argument from command line parameter --matrices-dir-install; aborting"
                exit 1
            fi
            shift 2 ;;
        --prefix)
            install_prefix=$(realpath ${2})
            shift 2 ;;
        -j)
            parallel_level=${2}
            shift 2 ;;
        -v|--verbose)
            verbose=true
            shift ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received: '${1}'; aborting";
            exit 1
            ;;
    esac
done

if [[ "${build_relocatable}" == true ]]; then
    rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,${rocm_rpath}"
fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
if [[ "${build_relocatable}" == true ]]; then
    export PATH=${rocm_path}/bin:${PATH}
else
    export PATH=${PATH}:/opt/rocm/bin
fi

#
# If matrices_dir_install has been set up then install matrices dir and exit.
#
if ! [[ "${matrices_dir_install}" == "" ]];then
    cmake -DCMAKE_CXX_COMPILER="${rocm_path}/bin/hipcc" -DCMAKE_C_COMPILER="${rocm_path}/bin/hipcc"  -DPROJECT_BINARY_DIR=${matrices_dir_install} -DCMAKE_MATRICES_DIR=${matrices_dir_install} -DROCM_PATH=${rocm_path} -DCMAKE_INSTALL_LIBDIR=lib -P ./cmake/ClientMatrices.cmake
    exit 0
fi

#
# If matrices_dir has been set up then check if it exists and it contains expected files.
# If it doesn't contain expected file, it will create them.
#
if ! [[ "${matrices_dir}" == "" ]];then
    if ! [ -e ${matrices_dir} ];then
        echo "Invalid dir from command line parameter --matrices-dir: ${matrices_dir}; aborting";
        exit 1
    fi

    # Let's 'reinstall' to the specified location to check if all good
    # Will be fast if everything already exists as expected.
    # This is to prevent any empty directory.
    cmake -DCMAKE_CXX_COMPILER="${rocm_path}/bin/hipcc" -DCMAKE_C_COMPILER="${rocm_path}/bin/hipcc" -DPROJECT_BINARY_DIR=${matrices_dir} -DCMAKE_MATRICES_DIR=${matrices_dir} -DROCM_PATH=${rocm_path} -DCMAKE_INSTALL_LIBDIR=lib -P ./cmake/ClientMatrices.cmake
fi

# #################################################
# prep
# #################################################

# ensure a clean build environment
if [ -f "${source_directory}/CMakeLists.txt" ] ; then
  source_dir="${source_directory}"
else
  printf "${cRED}Error: Source directory ${source_directory} does not contain CMakeLists.txt.${cRESET}\n"
  exit -1
fi

# Default cmake executable is called cmake
cmake_executable=cmake

# rhel9 does not have cmake3 but does have cmake
case "${ID}" in
  centos)
  cmake_executable=cmake3
  ;;
esac

# If user provides custom ${rocm_path} path for hcc it has lesser priority,
# but with hip-clang existing path has lesser priority to avoid use of installed clang++
if [[ "${build_hip_clang}" == true ]]; then
  export PATH=${rocm_path}/bin:${rocm_path}/hip/bin:${rocm_path}/llvm/bin:${PATH}
fi

# build type
cmake_common_options+=("-DGPU_TARGETS=${gpu_architecture}")
if [[ "${build_release}" == true ]]; then
  build_dir="${build_directory}/release"
  cmake_common_options+=("-DCMAKE_BUILD_TYPE=Release")
elif [[ "${build_release_debug}" == true ]]; then
  build_dir="${build_directory}/release-debug"
  cmake_common_options+=("-DCMAKE_BUILD_TYPE=RelWithDebInfo")
else
  build_dir="${build_directory}/debug"
  cmake_common_options+=("-DCMAKE_BUILD_TYPE=Debug")
fi

# remove build directory if it exists
if [[ -d "${build_dir}" ]]; then
  printf "${cbBLACK}${cRED}Removing existing build directory: ${cYELLOW}${build_dir}${cRESET}\n"
  # remove the build directory
  rm -rf ${build_dir}
fi

printf "${cbBLACK}${cGREEN}Creating project build directory in: ${cYELLOW}${build_dir}${cRESET}\n"
mkdir -p ${build_dir}

# #################################################
# configure & build
# #################################################

# address sanitizer
if [[ "${build_address_sanitizer}" == true ]]; then
  cmake_common_options+=("-DBUILD_ADDRESS_SANITIZER=ON")
fi

# warpsize32
if [[ "${build_warpsize32}" == true ]]; then
  cmake_common_options+=("-DROCGRAPH_USE_WARPSIZE_32=1")
fi

# memstat
if [[ "${build_memstat}" == true ]]; then
  cmake_common_options+=("-DBUILD_MEMSTAT=ON")
fi

# rocgraph_ILP64
if [[ "${build_rocgraph_ILP64}" == true ]]; then
  cmake_common_options+=("-DBUILD_ROCGRAPH_ILP64=ON")
fi

# no offload compress
if [[ "${build_with_offload_compress}" == true ]]; then
  cmake_common_options+=("-DBUILD_WITH_OFFLOAD_COMPRESS=ON")
else
  cmake_common_options+=("-DBUILD_WITH_OFFLOAD_COMPRESS=OFF")
fi

# freorg backward compatible support enable
if [[ "${build_freorg_bkwdcomp}" == true ]]; then
  cmake_common_options+=("-DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=ON")
else
  cmake_common_options+=("-DBUILD_FILE_REORG_BACKWARD_COMPATIBILITY=OFF")
fi

# code coverage
if [[ "${build_codecoverage}" == true ]]; then
    if [[ "${build_release}" == true ]]; then
        echo "Code coverage is disabled in Release mode, to enable code coverage select either Debug mode (-g | --debug) or RelWithDebInfo mode (-k | --relwithdebinfo); aborting";
        exit 1
    fi
    cmake_common_options+=("-DBUILD_CODE_COVERAGE=ON")
fi

# library type
if [[ "${build_static}" == true ]]; then
  cmake_common_options+=("-DBUILD_SHARED_LIBS=OFF")
fi

# clients
if [[ "${build_clients}" == true ]]; then
    cmake_client_options+=("-DBUILD_CLIENTS_TESTS=ON")
    #
    # Add matrices_dir if exists.
    #
    if ! [[ "${matrices_dir}" == "" ]];then
        cmake_client_options+=("-DCMAKE_MATRICES_DIR=${matrices_dir}")
    fi
fi

compiler="hcc"
if [[ "${build_hip_clang}" == true ]]; then
  compiler="${rocm_path}/bin/hipcc"
fi

# raft directory
if ! [[ "${raft_dir}" == "" ]]; then
  cmake_common_options+=("-DOVERRIDE_RAFT_SOURCE_DIR=${raft_dir}")
fi

# Build library with AMD toolchain because of existence of device kernels
if [[ "${build_relocatable}" == true ]]; then
  FC=gfortran CXX=${compiler} CC=${compiler} \
  ${cmake_executable} -S ${source_dir} -B ${build_dir} \
  ${cmake_common_options[@]} \
  ${cmake_client_options[@]} \
  -DCPACK_SET_DESTDIR=OFF \
  -DCMAKE_INSTALL_PREFIX=${install_prefix} \
  -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} \
  -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
  -DCMAKE_PREFIX_PATH="${rocm_path} ${rocm_path}/hcc ${rocm_path}/hip" \
  -DCMAKE_MODULE_PATH="${rocm_path}/lib/cmake/hip ${rocm_path}/hip/cmake" \
  -DROCM_DISABLE_LDCONFIG=ON \
  -DBUILD_VERBOSE=${verbose} \
  -DROCM_PATH="${rocm_path}"
else
  FC=gfortran CXX=${compiler} CC=${compiler} \
  ${cmake_executable} -S ${source_dir} -B ${build_dir} \
  ${cmake_common_options[@]} \
  ${cmake_client_options[@]} \
  -DCPACK_SET_DESTDIR=OFF \
  -DCMAKE_INSTALL_PREFIX=${install_prefix} \
  -DCPACK_PACKAGING_INSTALL_PREFIX=${rocm_path} \
  -DBUILD_VERBOSE=${verbose} \
  -DROCM_PATH="${rocm_path}"
fi
check_exit_code "$?"

# if VERBOSE is set to anything other than an empty string, make will be verbose
if [[ $verbose == 0 ]]; then
  nice ${cmake_executable} --build ${build_dir} --parallel ${parallel_level}
else
  VERBOSE=ON nice ${cmake_executable} --build ${build_dir} --parallel ${parallel_level}
fi
check_exit_code "$?"

# #################################################
# build package
# #################################################
if [[ "${build_package}" == true ]]; then
  if [[ $verbose == 0 ]]; then
    ${cmake_executable} --build ${build_dir} --target package
  else
    VERBOSE=ON ${cmake_executable} --build ${build_dir} --target package
  fi
  check_exit_code "$?"
fi

# #################################################
# install package
# #################################################
# Check if the user *may* believe they're installing into a "virtual
# environment." These checks are neither comprehensive nor reliable.
if [[ -v CONDA_ENV_PATH ]] ; then
    printf "${cbBLACK}${cRED}This script appears to be running in a Conda environment.\n${hSTAND}The packages are installed globally.${hunSTAND}${cRESET}"
elif [[ -v VIRTUAL_ENV ]] ; then
    printf "${cbBLACK}${cRED}This script appears to be running in a Python virtual environment.\n${hSTAND}The packages are installed globally.${hunSTAND}${cRESET}"
fi
# installing through package manager, which makes uninstalling easy
if [[ "${install_package}" == true ]]; then
  case "${ID}" in
    debian|ubuntu)
      elevate_if_not_root dpkg -i ${build_dir}/rocgraph[-\_]*.deb
    ;;
    centos|rhel)
      elevate_if_not_root yum -y localinstall ${build_dir}/rocgraph-*.rpm
    ;;
    fedora)
      elevate_if_not_root dnf install ${build_dir}/rocgraph-*.rpm
    ;;
    sles|opensuse-leap)
      elevate_if_not_root zypper -n --no-gpg-checks install ${build_dir}/rocgraph-*.rpm
    ;;
  esac
fi
