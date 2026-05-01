# File Manifest

## scripts/replay_realsense_hand_stickfigures_g1.py

Viewer-only Isaac Lab script.

Purpose:

- load Intel RealSense D435i plus MediaPipe hand trajectory JSON
- spawn a frozen Unitree G1 Inspire robot
- draw moving hand stick figures from the JSON landmarks
- verify coordinate-frame alignment before robot control

Final working options:

- --axis_map forward_z_flip_x
- --align_yaw_deg -90
- --hand_shape_scale 0.65
- --wrist_motion_scale 1.0
- --depth_motion_boost 1.0

## scripts/replay_realsense_g1_right_arm_fingers_with_stickfig.py

Robot-following Isaac Lab script with RealSense stick-figure overlay.

Purpose:

- load RealSense hand trajectory JSON
- draw the video hand stick figure target
- reuse the old VR split-architecture control path
- make the Unitree G1 right arm follow the video wrist trajectory
- make the Unitree G1 fingers curl from the video hand landmarks
- use the VR-trained wrist orientation model as a rough first palm-orientation attempt

Important implementation detail:

- RealSense JSON landmarks are converted into fake WebXR-style hand packets.
- The fake packet update must run every simulation loop.
- Without this, packets stays 0 and R_ok stays False, so the shoulder/elbow IK does not move.

Final working options:

- --video_axis_map forward_z_flip_x
- --video_align_yaw_deg -90
- --video_draw_hand_shape_scale 0.65
- --video_draw_wrist_motion_scale 1.0
- --video_draw_depth_motion_boost 1.0

## models/right_wrist_mapping_model_450_targeted.pt

Existing right-hand learned wrist orientation model.

Originally trained from WebXR calibration data.

Used here as a rough first attempt for video palm/wrist orientation.

## models/left_wrist_mapping_model_300.pt

Existing left-hand learned wrist orientation model.

Included because the bimanual split script can load both left and right wrist models.

## Known Issue Solved

The stick figure originally moved perfectly, and robot fingers/wrist joints moved, but the robot arm did not follow in space.

Cause:

- the old VR arm-position IK path was not receiving live packet data
- logs showed packets=0 and R_ok=False

Fix:

- create fake WebXR packets from the RealSense JSON
- call _video_update_fake_webxr_packet() every sim loop
- then the old arm-position controller receives valid video-derived hand targets

## Next Work

- train a video-specific palm orientation model
- clean the robot-following script
- improve finger curl retargeting
- test more RealSense videos

## scripts/replay_realsense_g1_right_arm_fingers_simpleboost.py

Current best RealSense video-to-robot replay script.

Purpose:

- replay RealSense MediaPipe hand trajectory JSON
- draw the RealSense stick-figure target
- drive the Unitree G1 arm/fingers from the video trajectory
- keep the successful coordinate mapping and yaw alignment
- add simple finger curl boost for better grip

Final working mapping:

- --video_axis_map forward_z_flip_x
- --video_align_yaw_deg -90

Final best finger setting:

- --simple_finger_boost
- --finger_curl_boost 1.5

Important implementation detail:

The simple finger boost must use the original retargeter's commanded target as the source, not the robot's current physical joint position.

Reason:

- using physical q_cur caused the robot hand to stay closed after gripping
- using the commanded target preserved opening behavior
- with boost 1.5, the robot opens/closes correctly and forms a good grip

Current status:

- 1.5 is the best current value
- good enough for now
- more experiments can be done later with other videos and grip types

### Best Current Run Command

Run inside the IsaacLab repo:

cd ~/IsaacLab_5

./isaaclab.sh -p scripts/replay_realsense_g1_right_arm_fingers_simpleboost.py \
  --device cpu \
  --video_axis_map forward_z_flip_x \
  --video_align_yaw_deg -90 \
  --video_draw_hand_shape_scale 0.65 \
  --video_draw_wrist_motion_scale 1.0 \
  --video_draw_depth_motion_boost 1.0 \
  --simple_finger_boost \
  --finger_curl_boost 1.5
