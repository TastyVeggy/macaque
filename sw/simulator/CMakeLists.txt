add_library(macaque_simulator STATIC
    lib/simulator.cpp
)

target_include_directories(macaque_simulator
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)

target_link_libraries(macaque_simulator PUBLIC macaque_common)

if(MACAQUE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
