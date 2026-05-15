add_library(macaque_common INTERFACE)

target_include_directories(macaque_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(test_instruction tests/test_instruction.cpp)
target_link_libraries(test_instruction PRIVATE macaque_common GTest::gtest_main)

add_test(NAME instruction_encoding COMMAND test_instruction)
