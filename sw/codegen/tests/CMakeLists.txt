add_executable(test_codegen
    test_load_weight_op.cpp
    test_load_bias_op.cpp
    test_load_input_op.cpp
    test_matmul_op.cpp
    test_activate_op.cpp
    test_store_op.cpp
    test_sync_op.cpp
    test_binary_emitter.cpp
    test_closed_loop.cpp
    test_tosa_to_macaque.cpp
)

target_link_libraries(test_codegen
    PRIVATE
        warnings
        macaque_codegen
        macaque_common
        macaque_simulator
        MLIRIR
        MLIRTosaDialect
        GTest::gtest_main
)

add_test(NAME codegen COMMAND test_codegen)
