add_executable(test_runtime
    test_program.cpp
    test_image.cpp
)

target_link_libraries(test_runtime
    PRIVATE
        warnings
        macaque_runtime
        macaque_common
        stb_image
        GTest::gtest_main
)

add_test(NAME runtime COMMAND test_runtime)
