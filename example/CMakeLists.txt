find_package(OpenCV REQUIRED COMPONENTS highgui imgproc videoio)

include_directories(${OpenCV_INCLUDE_DIR}
                    ${CMAKE_BINARY_DIR}
                    ${CMAKE_SOURCE_DIR})

add_executable(example_portrait_segmentation
        main.cc
        ERDNet.cc)

target_link_libraries(example_portrait_segmentation
                      ncnn
                      ${OpenCV_LIBS})