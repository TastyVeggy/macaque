add_executable(test_instruction test_instruction.cpp)

target_link_libraries(test_instruction
    PRIVATE
        warnings
        macaque_common
        GTest::gtest_main
)

add_test(NAME instruction_encoding COMMAND test_instruction)
