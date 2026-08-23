cmake_minimum_required(VERSION 3.28)
project(macaque_sw CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS NO)

option(MACAQUE_BUILD_TESTS "Build unit testing suite" ON)

set(MACAQUE_LLVM_BUILD "${CMAKE_SOURCE_DIR}/../third-party/llvm-project/build"
    CACHE PATH "Path to a built LLVM/MLIR")
if(EXISTS "${MACAQUE_LLVM_BUILD}")
    list(APPEND CMAKE_PREFIX_PATH "${MACAQUE_LLVM_BUILD}")
endif()

add_library(warnings INTERFACE)
target_compile_options(warnings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Werror
)

add_library(sanitizers INTERFACE)
target_compile_options(sanitizers INTERFACE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)
target_link_options(sanitizers INTERFACE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

if(MACAQUE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# subprojects to add
add_subdirectory(common)
add_subdirectory(codegen)
