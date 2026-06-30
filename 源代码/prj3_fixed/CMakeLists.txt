cmake_minimum_required(VERSION 3.10)
project(FisheyeNormalEstimation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 优先使用 OpenCV 官方 CMake 配置；若系统只安装了核心开发包，则回退到手动查找。
find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)

if(NOT OpenCV_FOUND)
    find_path(OPENCV_INCLUDE_DIR
        NAMES opencv2/core.hpp
        PATH_SUFFIXES opencv4
    )
    find_library(OPENCV_CORE_LIBRARY NAMES opencv_core)
    find_library(OPENCV_IMGCODECS_LIBRARY NAMES opencv_imgcodecs)
    find_library(OPENCV_IMGPROC_LIBRARY NAMES opencv_imgproc)

    if(NOT OPENCV_INCLUDE_DIR OR
       NOT OPENCV_CORE_LIBRARY OR
       NOT OPENCV_IMGCODECS_LIBRARY OR
       NOT OPENCV_IMGPROC_LIBRARY)
        message(FATAL_ERROR "OpenCV core/imgcodecs/imgproc development files were not found")
    endif()

    set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIR})
    set(OpenCV_LIBS
        ${OPENCV_CORE_LIBRARY}
        ${OPENCV_IMGCODECS_LIBRARY}
        ${OPENCV_IMGPROC_LIBRARY}
    )
endif()

add_executable(fisheye_normal
    src/main.cpp
    src/fisheye_utils.cpp
)

target_include_directories(fisheye_normal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(fisheye_normal PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(fisheye_normal PRIVATE /W4)
else()
    target_compile_options(fisheye_normal PRIVATE -Wall -Wextra -Wpedantic)
endif()
