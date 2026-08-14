# HuLing Deploy for Milk-V Duo S
# Cross-compile for RISC-V (musl) or ARM (glibc)
#
# Prerequisites:
#   1. Milk-V Duo S Buildroot SDK
#      git clone https://github.com/milkv-duo/duo-buildroot-sdk.git
#      export MILKV_SDK=/path/to/duo-buildroot-sdk
#
#   2. Build with:
#      # RISC-V:
#      source $MILKV_SDK/riscv64-buildroot/build/envsetup.sh
#      mkdir build-riscv && cd build-riscv
#      cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv.cmake
#      make
#
#      # ARM:
#      source $MILKV_SDK/arm-buildroot/build/envsetup.sh
#      mkdir build-arm && cd build-arm
#      cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-arm.cmake
#      make

cmake_minimum_required(VERSION 3.10)
project(huling_deploy C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# -----------------------------------------------------------
# Sources
# -----------------------------------------------------------
set(SOURCES
    # Core feature extraction (from MediaPipe-style 33 keypoints)
    ../huling_features.c
    # RandomForest 100-tree classifier (m2cgen generated)
    ../random_forest.c
    # Wrapper: standardize + classify
    ../random_forest_wrapper.c
)

# -----------------------------------------------------------
# Library target
# -----------------------------------------------------------
add_library(huling STATIC ${SOURCES})
target_include_directories(huling PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_options(huling PRIVATE
    -Os -Wall -ffast-math -fno-math-errno
)
target_link_libraries(huling m)

# -----------------------------------------------------------
# Standalone demo app
# (Feed keypoints from stdin/file, output class)
# -----------------------------------------------------------
add_executable(huling_demo main.c)
target_include_directories(huling_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(huling_demo huling m)

# -----------------------------------------------------------
# Verification test (compare against Python reference)
# -----------------------------------------------------------
add_executable(huling_test test_deploy_duos.c)
target_include_directories(huling_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(huling_test huling m)

# -----------------------------------------------------------
# UDP real-time inference server
# (Receives keypoints from PC, runs inference, prints results)
# -----------------------------------------------------------
add_executable(huling_server udp_server.c)
target_include_directories(huling_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(huling_server huling m)
