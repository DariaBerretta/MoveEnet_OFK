cmake_minimum_required(VERSION 3.16.0)
project(moveEnet_flow)

# Set C++ standard
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(YARP COMPONENTS os sig cv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(event-driven REQUIRED)
find_package(hpe-core REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

# Build edpr-april executable
# add_executable(edpr-april edpr-april.cpp)
# target_link_libraries(edpr-april
#     ${YARP_LIBRARIES}
#     ${OpenCV_LIBS}
#     event-driven::event-driven
#     hpe-core::hpe-core
# )

# Build extract_eros executable
# add_executable(extract_eros extract_eros.cpp)
# target_link_libraries(extract_eros
#     ${YARP_LIBRARIES}
#     ${OpenCV_LIBS}
#     event-driven::event-driven
# )

# Build moveEnet_flow executable (offline HPE)
add_executable(moveEnet_flow moveEnet_flow.cpp)
target_link_libraries(moveEnet_flow
    YARP::YARP_OS 
    YARP::YARP_sig
    YARP::YARP_cv
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)

# Installation
install(TARGETS moveEnet_flow
    RUNTIME DESTINATION bin
)
