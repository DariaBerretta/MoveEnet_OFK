cmake_minimum_required(VERSION 3.16.0)
project(moveEnet_flow)

# Find required packages
find_package(YARP COMPONENTS os sig cv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(event-driven REQUIRED)
find_package(hpe-core REQUIRED)

# Include directories
include_directories(
    ${YARP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Build moveEnet_flow executable (offline HPE)
add_executable(moveEnet_flow moveEnet_flow.cpp)
target_link_libraries(moveEnet_flow
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)

# Installation
install(TARGETS moveEnet_flow
    RUNTIME DESTINATION bin
)
