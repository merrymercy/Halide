add_executable(conv1d_nlc_process process.cpp)
halide_use_image_io(conv1d_nlc_process)

halide_generator(conv1d_nlc.generator SRCS conv1d_nlc_generator.cpp)
foreach(AUTO_SCHEDULE false true)
    if(${AUTO_SCHEDULE})
        set(LIB conv1d_nlc_auto_schedule)
    else()
        set(LIB conv1d_nlc)
    endif()
    halide_library_from_generator(${LIB}
                                  GENERATOR conv1d_nlc.generator
                                  GENERATOR_ARGS auto_schedule=${AUTO_SCHEDULE})
    target_link_libraries(conv1d_nlc_process PRIVATE ${LIB})
endforeach()
