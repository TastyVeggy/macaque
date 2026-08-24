add_executable(test_simulator test_simulator.cpp)

target_link_libraries(test_simulator
    PRIVATE
        warnings
        macaque_simulator
        GTest::gtest_main
)

add_test(NAME simulator COMMAND test_simulator)
