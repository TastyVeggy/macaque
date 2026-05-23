add_library(macaque_common INTERFACE)

target_include_directories(macaque_common INTERFACE include)

if(MACAQUE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
