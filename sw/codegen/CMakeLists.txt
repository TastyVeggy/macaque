find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include/macaque/Dialect)

add_mlir_dialect_library(macaque_codegen
    lib/Dialect/MacaqueDialect.cpp
    lib/Dialect/MacaqueOps.cpp
    lib/Target/BinaryEmitter.cpp
    lib/Target/DataEmitter.cpp
    lib/Conversion/DdrLayout.cpp
    lib/Conversion/TosaToMacaque.cpp

    DEPENDS
    MLIRMacaqueOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTosaDialect
    MLIRTransforms
)

target_compile_definitions(macaque_codegen PUBLIC ${LLVM_DEFINITIONS})

if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(macaque_codegen PUBLIC -fno-rtti)
endif()

target_include_directories(macaque_codegen
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
target_include_directories(macaque_codegen SYSTEM
    PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include"
)

target_link_libraries(macaque_codegen PRIVATE macaque_common)

if(MACAQUE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
