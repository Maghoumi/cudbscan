project(cudbscan)
cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=c++11 -fPIC -lusb-1.0 -lpthread)

set(CMAKE_BUILD_TYPE Release)

# To avoid OpenCV linking problems
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

#########################################################
# FIND CUDA
#########################################################
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -arch=compute_60")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21)

#########################################################
# Add sources and files
#########################################################
set(SOURCE_FILES
    Point2.hpp
    Timer.hpp
    DBSCANCPU.hpp
    cuDBSCAN.hpp
    cuDBSCAN.cu
    main.cpp)

cuda_add_executable(${PROJECT_NAME} ${SRC_LIST} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME}
${LIBS} ${CUDA_LIBRARIES})
