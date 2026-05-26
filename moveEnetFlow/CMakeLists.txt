cmake_minimum_required(VERSION 3.16.0)
project(moveEnet_flow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(YARP COMPONENTS os sig cv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(event-driven REQUIRED)
find_package(hpe-core REQUIRED)
find_package(OpenPose REQUIRED)
find_package(GFlags REQUIRED)

# Include directories
include_directories(
    ${YARP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenPose_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIR}
)

# Build moveEnet_flow executable (offline HPE)
# add_executable(moveEnet_flow moveEnet_flow.cpp)
# target_link_libraries(moveEnet_flow
#     ${YARP_LIBRARIES}
#     ${OpenCV_LIBS}
#     ev::event-driven
#     hpe-core::hpe-core
# )



# Build async no-idle Kalman pipeline
add_executable(moveEnetOFK_offline
    moveEnetOFK_offline.cpp
    utils/power_monitor.cpp
    utils/visualization_utils.cpp
)
target_link_libraries(moveEnetOFK_offline
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)

# Build async no-idle Kalman pipeline with in-binary MPJPE evaluation
add_executable(moveEnetOFK_offline_mpjpe
    moveEnetOFK_offline_mpjpe.cpp
    utils/power_monitor.cpp
    utils/visualization_utils.cpp
)
target_link_libraries(moveEnetOFK_offline_mpjpe
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)
# Build OpenPose offline pipeline (YARP-based Python detection, no OpenPose C++ library needed)
add_executable(OpenPose_offline
    OpenPose_offline.cpp
    utils/power_monitor.cpp
    utils/visualization_utils.cpp
)

target_link_libraries(OpenPose_offline
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
     ${OpenPose_LIBS}
    ${GFLAGS_LIBRARIES}
    ev::event-driven
    hpe-core::hpe-core
)


add_executable(YoloPose_offline
    YoloPose_offline.cpp
    utils/power_monitor.cpp
    utils/visualization_utils.cpp
)

target_link_libraries(YoloPose_offline
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)

add_executable(eventPointPose_offline
    eventPointPose_offline.cpp
    utils/power_monitor.cpp
    utils/visualization_utils.cpp
)

target_link_libraries(eventPointPose_offline
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)

#add_executable(eventPointPose_offline_preprocessed
#    eventPointPose_offline_preprocessed.cpp
#    utils/power_monitor.cpp
#    utils/visualization_utils.cpp
#)

#target_link_libraries(eventPointPose_offline_preprocessed
#    ${YARP_LIBRARIES}
#    ${OpenCV_LIBS}
#    ev::event-driven
#    hpe-core::hpe-core
#)

add_executable(moveEnetOFK_offline_power
    power_experiments/moveEnetOFK_offline_power.cpp
    power_experiments/power_pipeline_monitor.cpp
    utils/visualization_utils.cpp
)
target_include_directories(moveEnetOFK_offline_power PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/power_experiments
)
target_link_libraries(moveEnetOFK_offline_power
    ${YARP_LIBRARIES}
    ${OpenCV_LIBS}
    ev::event-driven
    hpe-core::hpe-core
)

# Build tDenseRNN offline pipeline for DHP19 raw event logs.
# add_executable(tDense_offline
#     tDense_offline.cpp
#     tdense_detector.cpp
# )
# target_link_libraries(tDense_offline
#     ${YARP_LIBRARIES}
#     ${OpenCV_LIBS}
#     ev::event-driven
#     hpe-core::hpe-core
# )

# add_executable(tdense_detector_tests
#     tdense_detector_tests.cpp
#     tdense_detector.cpp
# )
# target_link_libraries(tdense_detector_tests
#     ${YARP_LIBRARIES}
#     ${OpenCV_LIBS}
#     hpe-core::hpe-core
# )
