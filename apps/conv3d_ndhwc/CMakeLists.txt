add_executable(conv3d_ndhwc_process process.cpp)
halide_use_image_io(conv3d_ndhwc_process)

halide_generator(conv3d_ndhwc.generator SRCS conv3d_ndhwc_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    if(${AUTO_SCHEDULE})
        set(LIB conv3d_ndhwc_auto_schedule)
    else()
        set(LIB conv3d_ndhwc)
    endif()
    halide_library_from_generator(${LIB}
                                  GENERATOR conv3d_ndhwc.generator
                                  GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
    target_link_libraries(conv3d_ndhwc_process PRIVATE ${LIB})
endforeach()
