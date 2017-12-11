cmake_minimum_required(VERSION 3.8)
project(VisionTest)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp
#        StereoMatch.cpp
        opencv_contrib.cpp
        opencv_contrib.h
        tinydir.h
        StereoCalib.cpp
        )
add_executable(BinocularCalibration ${SOURCE_FILES})
target_link_libraries(BinocularCalibration ${OpenCV_LIBS})