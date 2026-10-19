cmake_minimum_required(VERSION 3.16)
project(optical_flow_sanity LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(YARP COMPONENTS os sig cv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(event-driven REQUIRED)
find_package(hpe-core REQUIRED)

add_executable(optical_flow_sanity ../optical_flow_sanity.cpp)
target_include_directories(optical_flow_sanity PRIVATE
    ${YARP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(optical_flow_sanity PRIVATE
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)
