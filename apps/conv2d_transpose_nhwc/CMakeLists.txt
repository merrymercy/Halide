add_executable(conv2d_transpose_nhwc_process process.cpp)
halide_use_image_io(conv2d_transpose_nhwc_process)

halide_generator(conv2d_transpose_nhwc.generator SRCS conv2d_transpose_nhwc_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    if(${AUTO_SCHEDULE})
        set(LIB conv2d_transpose_nhwc_auto_schedule)
    else()
        set(LIB conv2d_transpose_nhwc)
    endif()
    halide_library_from_generator(${LIB}
                                  GENERATOR conv2d_transpose_nhwc.generator
                                  GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
    target_link_libraries(conv2d_transpose_nhwc_process PRIVATE ${LIB})
endforeach()
