add_library(stb_image INTERFACE)
target_include_directories(stb_image SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")

add_library(macaque_runtime STATIC
    lib/SerialTransport.cpp
    lib/Protocol.cpp
    lib/Program.cpp
    lib/Image.cpp
    lib/Quantize.cpp
    lib/Device.cpp
)

target_include_directories(macaque_runtime
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(macaque_runtime
    PUBLIC macaque_common
    PRIVATE warnings stb_image
)

add_executable(macaque tools/macaque.cpp)
target_link_libraries(macaque PRIVATE warnings macaque_runtime)

install(TARGETS macaque RUNTIME DESTINATION bin)

if(MACAQUE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

option(MACAQUE_BUILD_EXAMPLES "Build sw/runtime/examples" ON)
if(MACAQUE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
