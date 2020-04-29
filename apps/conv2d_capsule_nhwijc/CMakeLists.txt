add_executable(conv2d_capsule_nhwijc_process process.cpp)
halide_use_image_io(conv2d_capsule_nhwijc_process)

halide_generator(conv2d_capsule_nhwijc.generator SRCS conv2d_capsule_nhwijc_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    if(${AUTO_SCHEDULE})
        set(LIB conv2d_capsule_nhwijc_auto_schedule)
    else()
        set(LIB conv2d_capsule_nhwijc)
    endif()
    halide_library_from_generator(${LIB}
                                  GENERATOR conv2d_capsule_nhwijc.generator
                                  GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
    target_link_libraries(conv2d_capsule_nhwijc_process PRIVATE ${LIB})
endforeach()
