import argparse
import json
import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--json",
    type=str,
    default="~/Downloads/20260430_133659_soft_kinematic_ready_realsense_d435i_mediapipe_hand_trajectory_v1.json",
)
parser.add_argument("--loop", action="store_true")
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument("--fps_override", type=float, default=0.0)

parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--wrist_motion_scale", type=float, default=None)
parser.add_argument("--hand_shape_scale", type=float, default=None)
parser.add_argument("--depth_motion_boost", type=float, default=1.0)
parser.add_argument("--x_offset", type=float, default=0.0)
parser.add_argument("--y_offset", type=float, default=0.0)
parser.add_argument("--z_offset", type=float, default=0.0)

parser.add_argument("--wrist_jump_limit", type=float, default=0.12)
parser.add_argument("--wrist_smooth_alpha", type=float, default=0.25)

parser.add_argument(
    "--landmark_source",
    type=str,
    default="auto",
    choices=["auto", "stabilized", "raw"],
    help="auto uses stabilized/kinematic landmarks if present, otherwise raw landmarks_3d.",
)

parser.add_argument("--free_camera_space", action="store_true")
parser.add_argument("--free_origin_x", type=float, default=0.35)
parser.add_argument("--free_origin_y", type=float, default=0.0)
parser.add_argument("--free_origin_z", type=float, default=1.15)
parser.add_argument("--align_yaw_deg", type=float, default=0.0, help="Rotate replay motion around Isaac Z after camera mapping.")
parser.add_argument("--view", type=str, default="diag", choices=["diag", "side_neg_y", "side_pos_y", "front"])
parser.add_argument(
    "--axis_map",
    type=str,
    default="cv_forward_x",
    choices=[
        "cv_forward_x",
        "webxr_like",
        "forward_y",
        "forward_neg_y",
        "forward_z",
        "forward_z_flip_x",
        "forward_neg_x",
    ],
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.wrist_motion_scale is None:
    args_cli.wrist_motion_scale = args_cli.scale

if args_cli.hand_shape_scale is None:
    args_cli.hand_shape_scale = args_cli.scale

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG


LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

HAND_BONES = [
    ("WRIST", "THUMB_CMC"),
    ("THUMB_CMC", "THUMB_MCP"),
    ("THUMB_MCP", "THUMB_IP"),
    ("THUMB_IP", "THUMB_TIP"),

    ("WRIST", "INDEX_FINGER_MCP"),
    ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
    ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),

    ("WRIST", "MIDDLE_FINGER_MCP"),
    ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
    ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),

    ("WRIST", "RING_FINGER_MCP"),
    ("RING_FINGER_MCP", "RING_FINGER_PIP"),
    ("RING_FINGER_PIP", "RING_FINGER_DIP"),
    ("RING_FINGER_DIP", "RING_FINGER_TIP"),

    ("WRIST", "PINKY_MCP"),
    ("PINKY_MCP", "PINKY_PIP"),
    ("PINKY_PIP", "PINKY_DIP"),
    ("PINKY_DIP", "PINKY_TIP"),
]


def setup_debug_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        print("[VIZ] debug draw enabled")
        return draw
    except Exception as e:
        print("[VIZ] debug draw unavailable:", e)
        return None


def safe_clear_draw(draw):
    if draw is None:
        return
    try:
        draw.clear_points()
    except Exception:
        pass
    try:
        draw.clear_lines()
    except Exception:
        pass


def tup(v):
    return tuple(float(x) for x in v.detach().cpu().tolist())


def draw_axis_frame(draw, origin, scale=0.12):
    if draw is None:
        return

    origin = origin.clone()
    x_end = origin + torch.tensor([scale, 0.0, 0.0], device=origin.device)
    y_end = origin + torch.tensor([0.0, scale, 0.0], device=origin.device)
    z_end = origin + torch.tensor([0.0, 0.0, scale], device=origin.device)

    draw.draw_lines(
        [tup(origin), tup(origin), tup(origin)],
        [tup(x_end), tup(y_end), tup(z_end)],
        [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.2, 1.0, 1.0),
        ],
        [5.0, 5.0, 5.0],
    )


def draw_hand(draw, points, color):
    if draw is None or points is None:
        return

    draw_pts = []
    for name in LANDMARK_NAMES:
        p = points.get(name)
        if p is not None:
            draw_pts.append(tup(p))

    if draw_pts:
        draw.draw_points(draw_pts, [color] * len(draw_pts), [7.0] * len(draw_pts))

    starts, ends = [], []
    for a, b in HAND_BONES:
        pa = points.get(a)
        pb = points.get(b)
        if pa is not None and pb is not None:
            starts.append(tup(pa))
            ends.append(tup(pb))

    if starts:
        draw.draw_lines(starts, ends, [color] * len(starts), [2.5] * len(starts))


def camera_to_isaac(p, device):
    """
    RealSense/color optical frame:
      camera x = right in image
      camera y = down in image
      camera z = forward from camera

    Testable mappings:
      cv_forward_x:  [ z, -x, -y]  # current guess
      webxr_like:    [-z, -x,  y]  # similar to our old WebXR visual convention
      forward_y:     [-x,  z, -y]
      forward_neg_y: [-x, -z, -y]
      forward_z:     [-x, -y,  z]
      forward_neg_x: [-z, -x, -y]
    """
    x, y, z = p

    if args_cli.axis_map == "cv_forward_x":
        out = [z, -x, -y]
    elif args_cli.axis_map == "webxr_like":
        out = [-z, -x, y]
    elif args_cli.axis_map == "forward_y":
        out = [-x, z, -y]
    elif args_cli.axis_map == "forward_neg_y":
        out = [-x, -z, -y]
    elif args_cli.axis_map == "forward_z":
        out = [-x, -y, z]
    elif args_cli.axis_map == "forward_z_flip_x":
        out = [x, -y, z]
    elif args_cli.axis_map == "forward_neg_x":
        out = [-z, -x, -y]
    else:
        out = [z, -x, -y]

    return torch.tensor(out, dtype=torch.float32, device=device)


def valid_point(p):
    return (
        isinstance(p, (list, tuple))
        and len(p) == 3
        and all(v is not None for v in p)
    )


def get_hand_present(frame, hand):
    return bool(frame.get(f"{hand}_hand_present", False))


def get_landmark_map(frame, hand):
    if not get_hand_present(frame, hand):
        return None

    raw_key = f"{hand}_landmarks_3d"
    stabilized_keys = [
        f"{hand}_stabilized_landmarks_3d",
        f"{hand}_kinematic_landmarks_3d",
        f"{hand}_landmarks_3d_stabilized",
        f"{hand}_landmarks_3d_kinematic",
    ]

    if args_cli.landmark_source == "raw":
        lms = frame.get(raw_key)
        return lms if isinstance(lms, dict) else None

    if args_cli.landmark_source in ["auto", "stabilized"]:
        for key in stabilized_keys:
            lms = frame.get(key)
            if isinstance(lms, dict):
                return lms

    lms = frame.get(raw_key)
    return lms if isinstance(lms, dict) else None


def find_first_wrist_origin(frames, hand, device):
    for frame in frames:
        lms = get_landmark_map(frame, hand)
        if not lms:
            continue

        wrist = lms.get("WRIST")
        if valid_point(wrist):
            return camera_to_isaac(wrist, device)

    return None


last_good_wrist = {
    "left": None,
    "right": None,
}

smooth_wrist = {
    "left": None,
    "right": None,
}


def get_filtered_wrist(raw_wrist, hand):
    if last_good_wrist[hand] is not None:
        jump = torch.linalg.norm(raw_wrist - last_good_wrist[hand])
        if float(jump) > args_cli.wrist_jump_limit:
            raw_wrist = last_good_wrist[hand]
        else:
            last_good_wrist[hand] = raw_wrist
    else:
        last_good_wrist[hand] = raw_wrist

    if smooth_wrist[hand] is None:
        smooth_wrist[hand] = raw_wrist
    else:
        a = args_cli.wrist_smooth_alpha
        smooth_wrist[hand] = (1.0 - a) * smooth_wrist[hand] + a * raw_wrist

    return smooth_wrist[hand]



def rotate_yaw_z(v, yaw_deg):
    """
    Rotate vector around Isaac Z axis.
    Used to align video/camera motion direction to robot action space.
    """
    if abs(yaw_deg) < 1e-6:
        return v

    yaw = torch.tensor(yaw_deg * 3.141592653589793 / 180.0, dtype=torch.float32, device=v.device)
    c = torch.cos(yaw)
    ss = torch.sin(yaw)

    x = v[..., 0].clone()
    y = v[..., 1].clone()
    z = v[..., 2].clone()

    out = v.clone()
    out[..., 0] = c * x - ss * y
    out[..., 1] = ss * x + c * y
    out[..., 2] = z
    return out


def transformed_landmarks(frame, hand, anchor_robot_wrist, origin_isaac, device):
    lms = get_landmark_map(frame, hand)
    if lms is None or origin_isaac is None:
        return None

    raw = {}
    for name in LANDMARK_NAMES:
        p = lms.get(name)
        raw[name] = camera_to_isaac(p, device) if valid_point(p) else None

    raw_wrist = raw.get("WRIST")
    if raw_wrist is None:
        return None

    wrist_filtered = get_filtered_wrist(raw_wrist, hand)

    offset = torch.tensor(
        [args_cli.x_offset, args_cli.y_offset, args_cli.z_offset],
        dtype=torch.float32,
        device=device,
    )

    # Motion relative to first detected video wrist.
    motion = wrist_filtered - origin_isaac
    motion = motion.clone()
    motion = rotate_yaw_z(motion, args_cli.align_yaw_deg)

    # In cv_forward_x / webxr_like / forward_neg_x modes, depth mostly maps to Isaac X.
    # This is a debug boost only, useful for seeing RealSense forward/back motion.
    motion[0] = motion[0] * args_cli.depth_motion_boost

    if args_cli.free_camera_space:
        free_origin = torch.tensor(
            [args_cli.free_origin_x, args_cli.free_origin_y, args_cli.free_origin_z],
            dtype=torch.float32,
            device=device,
        )
        world_wrist = free_origin + motion * args_cli.wrist_motion_scale + offset
    else:
        world_wrist = anchor_robot_wrist + motion * args_cli.wrist_motion_scale + offset

    out = {}
    for name in LANDMARK_NAMES:
        p = raw.get(name)
        if p is None:
            out[name] = None
            continue

        # Keep hand shape relative to wrist. This can be scaled independently from wrist trajectory.
        rel = p - raw_wrist
        rel = rotate_yaw_z(rel, args_cli.align_yaw_deg)
        out[name] = world_wrist + rel * args_cli.hand_shape_scale

    return out


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


json_path = Path(args_cli.json).expanduser()
if not json_path.exists():
    raise FileNotFoundError(f"JSON not found: {json_path}")

data = json.loads(json_path.read_text())
metadata = data.get("metadata", {})
diagnostics = metadata.get("diagnostics", {})
kinematic = metadata.get("kinematic_stabilization", {})
frames = data.get("frames", [])

if not frames:
    raise RuntimeError("JSON has no frames")

fps = args_cli.fps_override if args_cli.fps_override > 0 else float(metadata.get("fps", 30.0))
dt_frame = 1.0 / fps

print("")
print("================================================")
print("REALSENSE VIDEO JSON HAND STICK FIGURE REPLAY")
print("================================================")
print("JSON:", json_path)
print("schema:", metadata.get("schema"))
print("schema_version:", metadata.get("schema_version"))
print("fps:", fps)
print("frames:", len(frames))
print("coord_frame:", metadata.get("coord_frame"))
print("position_units:", metadata.get("position_units"))
print("landmark_source:", args_cli.landmark_source)
print("wrist_motion_scale:", args_cli.wrist_motion_scale)
print("hand_shape_scale:", args_cli.hand_shape_scale)
print("depth_motion_boost:", args_cli.depth_motion_boost)
print("axis_map:", args_cli.axis_map)
print("align_yaw_deg:", args_cli.align_yaw_deg)
print("kinematic enabled:", kinematic.get("enabled"))
print("kinematic method:", kinematic.get("method"))
print("diagnostics:", diagnostics)
print("")
print("Robot is spawned frozen.")
print("Hands are drawn from RealSense video JSON landmarks.")
print("No robot control yet.")
print("================================================")
print("")


sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
if args_cli.view == "diag":
    sim.set_camera_view([2.7, -2.7, 1.6], [0.15, 0.0, 1.05])
elif args_cli.view == "side_neg_y":
    sim.set_camera_view([0.35, -3.0, 1.3], [0.35, 0.0, 1.1])
elif args_cli.view == "side_pos_y":
    sim.set_camera_view([0.35, 3.0, 1.3], [0.35, 0.0, 1.1])
elif args_cli.view == "front":
    sim.set_camera_view([3.0, 0.0, 1.3], [0.35, 0.0, 1.1])

scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
sim.reset()

robot = scene["robot"]
draw = setup_debug_draw()

left_body_ids, left_body_names = robot.find_bodies(["left_wrist_yaw_link"])
right_body_ids, right_body_names = robot.find_bodies(["right_wrist_yaw_link"])

left_body_id = left_body_ids[0]
right_body_id = right_body_ids[0]

print("left wrist body:", left_body_id, left_body_names)
print("right wrist body:", right_body_id, right_body_names)

default_q = robot.data.default_joint_pos.clone()
zero_v = torch.zeros_like(default_q)
robot.write_joint_state_to_sim(default_q, zero_v)
robot.set_joint_position_target(default_q)

left_origin = find_first_wrist_origin(frames, "left", args_cli.device)
right_origin = find_first_wrist_origin(frames, "right", args_cli.device)

print("left video origin:", None if left_origin is None else left_origin.detach().cpu().tolist())
print("right video origin:", None if right_origin is None else right_origin.detach().cpu().tolist())

frame_idx = max(0, min(args_cli.start_frame, len(frames) - 1))
next_frame_time = time.time()
last_log_time = 0.0

while simulation_app.is_running():
    scene.update(sim.get_physics_dt())

    robot.set_joint_position_target(default_q)
    scene.write_data_to_sim()
    sim.step()

    now = time.time()

    if now >= next_frame_time:
        frame = frames[frame_idx]

        safe_clear_draw(draw)

        left_wrist_robot = robot.data.body_link_pos_w[0, left_body_id]
        right_wrist_robot = robot.data.body_link_pos_w[0, right_body_id]

        draw_axis_frame(draw, left_wrist_robot, scale=0.10)
        draw_axis_frame(draw, right_wrist_robot, scale=0.10)

        left_points = transformed_landmarks(
            frame,
            "left",
            left_wrist_robot,
            left_origin,
            args_cli.device,
        )
        right_points = transformed_landmarks(
            frame,
            "right",
            right_wrist_robot,
            right_origin,
            args_cli.device,
        )

        draw_hand(draw, left_points, color=(0.0, 0.8, 1.0, 1.0))
        draw_hand(draw, right_points, color=(1.0, 0.45, 0.0, 1.0))

        if now - last_log_time > 1.0:
            last_log_time = now
            print(
                f"[REPLAY] frame={frame_idx}/{len(frames)-1} "
                f"t={frame.get('timestamp_sec')} "
                f"L={frame.get('left_hand_present')} "
                f"R={frame.get('right_hand_present')} "
                f"L_valid={frame.get('left_valid_landmark_count')} "
                f"R_valid={frame.get('right_valid_landmark_count')}"
            )

        frame_idx += 1

        if frame_idx >= len(frames):
            if args_cli.loop:
                frame_idx = 0
            else:
                print("[DONE] replay finished")
                break

        next_frame_time = now + dt_frame

simulation_app.close()
