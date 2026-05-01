# Isaac Video Trajectory Replay

This repo contains Isaac Lab scripts for replaying hand trajectories extracted from Intel RealSense D435i bag recordings.

The current pipeline loads a RealSense plus MediaPipe hand trajectory JSON, spawns a Unitree G1 Inspire robot, draws the video hand stick figure, and can now make the robot arm/fingers follow the video hand.

## Main Scripts

### scripts/replay_realsense_hand_stickfigures_g1.py

Viewer-only script.

It spawns a frozen Unitree G1 and draws the RealSense hand stick figure from the JSON. No robot control happens in this script.

Final working viewer mapping:

- axis_map: forward_z_flip_x
- align_yaw_deg: -90

### scripts/replay_realsense_g1_right_arm_fingers_with_stickfig.py

Robot-following script with stick-figure overlay.

It reuses the previous VR split-architecture logic, but replaces live WebXR input with RealSense JSON landmarks.

It draws the orange/cyan RealSense stick figure and also drives the robot arm/fingers.

Current behavior:

- right wrist position follows the RealSense video trajectory
- right fingers follow/curl from the video landmarks
- palm orientation uses the existing VR-trained learned wrist model as a rough first attempt
- the stick figure is drawn at the same time so robot-vs-target error is visible

## Required Local Files

The robot-following script uses these learned wrist models:

- models/right_wrist_mapping_model_450_targeted.pt
- models/left_wrist_mapping_model_300.pt

In the current IsaacLab working setup, these are also expected at:

- ~/IsaacLab_5/right_wrist_mapping_model_450_targeted.pt
- ~/IsaacLab_5/left_wrist_mapping_model_300.pt

## Default Input JSON

The scripts expect this RealSense JSON by default:

~/Downloads/20260430_133659_soft_kinematic_ready_realsense_d435i_mediapipe_hand_trajectory_v1.json

## Final Working Stick-Figure Viewer Run

Run inside IsaacLab:

cd ~/IsaacLab_5

./isaaclab.sh -p scripts/replay_realsense_hand_stickfigures_g1.py \
  --device cpu \
  --axis_map forward_z_flip_x \
  --align_yaw_deg -90 \
  --view side_neg_y \
  --wrist_smooth_alpha 1.0 \
  --wrist_jump_limit 999 \
  --hand_shape_scale 0.65 \
  --wrist_motion_scale 1.0 \
  --depth_motion_boost 1.0 \
  --loop

## Final Working Robot-Following Run

cd ~/IsaacLab_5

./isaaclab.sh -p scripts/replay_realsense_g1_right_arm_fingers_with_stickfig.py \
  --device cpu \
  --video_axis_map forward_z_flip_x \
  --video_align_yaw_deg -90 \
  --video_draw_hand_shape_scale 0.65 \
  --video_draw_wrist_motion_scale 1.0 \
  --video_draw_depth_motion_boost 1.0

## Coordinate Fixes

The RealSense JSON uses the color optical frame:

- camera x = right in image
- camera y = down in image
- camera z = forward from camera

The mapping that made the local hand motion and thumb side correct was:

out = [x, -y, z]

This is exposed as:

--axis_map forward_z_flip_x

Then the whole replay trajectory needed to be rotated into the robot action-space frame:

--align_yaw_deg -90

Final interpretation:

- forward_z_flip_x fixes the RealSense-to-Isaac local hand coordinate mapping
- align_yaw_deg -90 aligns the trajectory with the Unitree G1 robot frame

## Robot Movement Bug and Fix

At first, the stick figure moved perfectly and the robot fingers/wrist joints moved, but the robot arm/elbow did not follow the target in space.

The logs showed packets=0 and R_ok=False, which meant the old VR arm-position IK path was not receiving a valid live hand-position packet. The learned wrist orientation was updating, but the shoulder/elbow IK was not active.

The fix was to create a fake WebXR packet every simulation loop from the RealSense JSON frame, then feed that packet into the old VR control path.

After forcing _video_update_fake_webxr_packet() to run every loop, packets started reaching the old position controller and the robot arm began following the RealSense stick-figure trajectory.

## RealSense Data Quality

The RealSense D435i JSON was much better than the old stereo-camera output.

Useful diagnostics from the successful JSON:

- right hand detections: 241 frames
- right average valid 3D landmarks: about 20.45 / 21
- right median wrist depth: about 0.638 m
- right median palm width: about 0.060 m
- right wrist jump p90: about 0.017 m
- right wrist jump max: about 0.046 m

## Next Steps

1. Clean up the robot-following script.
2. Make right-arm replay more stable.
3. Train a video-landmarks-to-G1-palm-orientation model instead of using the VR-trained model.
4. Improve finger curl retargeting.
5. Add left-hand video replay after collecting a left-hand RealSense JSON.
