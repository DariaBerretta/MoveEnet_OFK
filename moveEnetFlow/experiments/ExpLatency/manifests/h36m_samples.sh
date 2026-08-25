#!/usr/bin/env bash

# Latency experiment manifest for paired eH36M/H36M sequences.
#
# Fields:
# sample_id | subject | sequence | camera | order_id | event_path | rgb_path
#
# Balanced execution orders:
#
# H0: MoveNet     -> MoveEnetOFK -> OpenPose -> YOLO
# H1: MoveEnetOFK -> YOLO        -> MoveNet  -> OpenPose
# H2: YOLO        -> OpenPose    -> MoveEnetOFK -> MoveNet
# H3: OpenPose    -> MoveNet     -> YOLO -> MoveEnetOFK
#
# Each order occurs exactly 3 times.
# S9:  H0=2, H1=2, H2=1, H3=1
# S11: H0=1, H1=1, H2=2, H3=2

H36M_SAMPLES=(

# ----------------------------------------------------------------------
# Discussion - natural upper-body gesturing
# ----------------------------------------------------------------------
"H01|S9|Discussion_1|cam2|H0|/data/eh36m_testing_set_S9S11/events/cam2_S9_Discussion_1/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Discussion_1.mp4"

"H07|S11|Discussion_2|cam2|H2|/data/eh36m_testing_set_S9S11/events/cam2_S11_Discussion_2/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S11_Discussion_2.mp4"


# ----------------------------------------------------------------------
# Eating - localized upper-body movement
# ----------------------------------------------------------------------
"H02|S9|Eating|cam2|H1|/data/eh36m_testing_set_S9S11/events/cam2_S9_Eating/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Eating.mp4"

"H08|S11|Eating_1|cam2|H3|/data/eh36m_testing_set_S9S11/events/cam2_S11_Eating_1/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S11_Eating_1.mp4"


# ----------------------------------------------------------------------
# Sitting - relatively low-motion condition
# ----------------------------------------------------------------------
"H03|S9|Sitting_1|cam2|H2|/data/eh36m_testing_set_S9S11/events/cam2_S9_Sitting_1/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Sitting_1.mp4"

"H09|S11|Sitting|cam2|H0|/data/eh36m_testing_set_S9S11/events/cam2_S11_Sitting/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S11_Sitting.mp4"


# ----------------------------------------------------------------------
# SittingDown - large pose transition
# ----------------------------------------------------------------------
"H04|S9|SittingDown|cam2|H3|/data/eh36m_testing_set_S9S11/events/cam2_S9_SittingDown/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S9_SittingDown.mp4"

"H10|S11|SittingDown_1|cam2|H1|/data/eh36m_testing_set_S9S11/events/cam2_S11_SittingDown_1/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S11_SittingDown_1.mp4"


# ----------------------------------------------------------------------
# Walking - periodic whole-body locomotion
# ----------------------------------------------------------------------
"H05|S9|Walking_1|cam2|H0|/data/eh36m_testing_set_S9S11/events/cam2_S9_Walking_1/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S9_Walking_1.mp4"

"H11|S11|Walking|cam2|H2|/data/eh36m_testing_set_S9S11/events/cam2_S11_Walking/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S11_Walking.mp4"


# ----------------------------------------------------------------------
# WalkDog - locomotion plus upper-body motion
# ----------------------------------------------------------------------
"H06|S9|WalkDog|cam2|H1|/data/eh36m_testing_set_S9S11/events/cam2_S9_WalkDog/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S9_WalkDog.mp4"

"H12|S11|WalkDog_1|cam2|H3|/data/eh36m_testing_set_S9S11/events/cam2_S11_WalkDog_1/ch0dvs/data.log|/data/eh36m_testing_set_S9S11/rgb/cam2_S11_WalkDog_1.mp4"

)
