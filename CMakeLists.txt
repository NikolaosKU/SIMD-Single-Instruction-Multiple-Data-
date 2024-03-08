cmake_minimum_required(VERSION 3.2)
project(bdap-assignment2-simd)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## default by prof
## set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -mavx -mavx2")

## for gcc (I used for remote for linux 'perf' profiling which is a too complicated with clang localy, regret having bought mac)
## set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -mavx -mavx2 -mfma -fno-tree-vectorize -fopt-info-vec")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O1 -fno-vectorize -fno-slp-vectorize -mavx -mavx2 -mfma -fopt-info-vec -fopt-info-vec-missed")


## for clang
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O1 -fno-vectorize -fno-slp-vectorize -mavx -mavx2 -mfma -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(BDAP_ASSIGNMENT2_INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(BDAP_ASSIGNMENT2_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(BDAP_ASSIGNMENT2_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

include_directories(${BDAP_ASSIGNMENT2_INSTALL_INCLUDE_DIR})

add_subdirectory(src)
