#!/usr/bin/env bash

# Latency experiment manifest for the DHP19 test subset.
#
# The locally available subset contains Session 1 only:
#
# 1 - Left arm abduction       [upper]
# 2 - Right arm abduction      [upper]
# 3 - Left leg abduction       [lower]
# 4 - Right leg abduction      [lower]
# 5 - Left arm bicep curl      [upper]
# 6 - Right arm bicep curl     [upper]
# 7 - Left leg knee lift       [lower]
# 8 - Right leg knee lift      [lower]
#
# Fields:
# sample_id | subject | sequence | camera | motion_class |
# motion_name | order_id | event_path
#
# Balanced execution orders:
#
# D0: MoveNet        -> MoveEnetOFK   -> EventPointPose
# D1: MoveNet        -> EventPointPose -> MoveEnetOFK
# D2: MoveEnetOFK    -> MoveNet       -> EventPointPose
# D3: MoveEnetOFK    -> EventPointPose -> MoveNet
# D4: EventPointPose -> MoveNet       -> MoveEnetOFK
# D5: EventPointPose -> MoveEnetOFK   -> MoveNet
#
# Each order occurs exactly twice:
# one ch2dvs sample and one ch3dvs sample.

DHP19_SAMPLES=(

# D0
"D01|S13|S13_1_1|ch2dvs|upper|left_arm_abduction|D0|/data/dhp19_testing_set_S13toS17/S13_1_1/ch2dvs/data.log"
"D02|S14|S14_1_3|ch3dvs|lower|left_leg_abduction|D0|/data/dhp19_testing_set_S13toS17/S14_1_3/ch3dvs/data.log"

# D1
"D03|S15|S15_1_4|ch2dvs|lower|right_leg_abduction|D1|/data/dhp19_testing_set_S13toS17/S15_1_4/ch2dvs/data.log"
"D04|S16|S16_1_2|ch3dvs|upper|right_arm_abduction|D1|/data/dhp19_testing_set_S13toS17/S16_1_2/ch3dvs/data.log"

# D2
"D05|S17|S17_1_5|ch2dvs|upper|left_arm_bicep_curl|D2|/data/dhp19_testing_set_S13toS17/S17_1_5/ch2dvs/data.log"
"D06|S13|S13_1_7|ch3dvs|lower|left_leg_knee_lift|D2|/data/dhp19_testing_set_S13toS17/S13_1_7/ch3dvs/data.log"

# D3
"D07|S14|S14_1_8|ch2dvs|lower|right_leg_knee_lift|D3|/data/dhp19_testing_set_S13toS17/S14_1_8/ch2dvs/data.log"
"D08|S15|S15_1_6|ch3dvs|upper|right_arm_bicep_curl|D3|/data/dhp19_testing_set_S13toS17/S15_1_6/ch3dvs/data.log"

# D4
"D09|S16|S16_1_7|ch2dvs|lower|left_leg_knee_lift|D4|/data/dhp19_testing_set_S13toS17/S16_1_7/ch2dvs/data.log"
"D10|S17|S17_1_1|ch3dvs|upper|left_arm_abduction|D4|/data/dhp19_testing_set_S13toS17/S17_1_1/ch3dvs/data.log"

# D5
"D11|S13|S13_1_2|ch2dvs|upper|right_arm_abduction|D5|/data/dhp19_testing_set_S13toS17/S13_1_2/ch2dvs/data.log"
"D12|S14|S14_1_4|ch3dvs|lower|right_leg_abduction|D5|/data/dhp19_testing_set_S13toS17/S14_1_4/ch3dvs/data.log"

)
