# Toolchain for Milk-V Duo S (RISC-V 64-bit, musl libc)
# Requires: duo-buildroot-sdk with riscv64 toolchain built

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Adjust these paths to match your SDK installation
set(SDK_ROOT "$ENV{MILKV_SDK}" CACHE PATH "Milk-V Duo Buildroot SDK root")

set(CROSS_PREFIX "${SDK_ROOT}/host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-")

set(CMAKE_C_COMPILER   "${CROSS_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${CROSS_PREFIX}g++")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Optimize for Duo S's C906 core
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc -mabi=lp64d -Os")
