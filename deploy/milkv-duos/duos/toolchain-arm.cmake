# Toolchain for Milk-V Duo S (ARM Cortex-A53 mode)
# Requires: duo-buildroot-sdk with arm toolchain built

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(SDK_ROOT "$ENV{MILKV_SDK}" CACHE PATH "Milk-V Duo Buildroot SDK root")

set(CROSS_PREFIX "${SDK_ROOT}/host-tools/gcc/arm-buildroot-linux-gnueabihf/bin/arm-buildroot-linux-gnueabihf-")

set(CMAKE_C_COMPILER   "${CROSS_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${CROSS_PREFIX}g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Optimize for Cortex-A53
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a53 -mfpu=neon-fp-armv8 -mfloat-abi=hard -Os")
