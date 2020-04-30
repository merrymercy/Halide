add_executable(depthwise_conv2d_nhwc_process process.cpp)
halide_use_image_io(depthwise_conv2d_nhwc_process)

halide_generator(depthwise_conv2d_nhwc.generator SRCS depthwise_conv2d_nhwc_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    if(${AUTO_SCHEDULE})
        set(LIB depthwise_conv2d_nhwc_auto_schedule)
    else()
        set(LIB depthwise_conv2d_nhwc)
    endif()
    halide_library_from_generator(${LIB}
                                  GENERATOR depthwise_conv2d_nhwc.generator
                                  GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
    target_link_libraries(depthwise_conv2d_nhwc_process PRIVATE ${LIB})
endforeach()
