# --- SET YOUR MASTER MACHINE IP IN THE 'master_ip' VARIABLE BELOW TO RECEIVE TRACKING DATA (currently set to 10.10.10.3) ---
"""
SuperMesh — headless_script.py (Node)
Runs on each Beelink. Two phases:
  1. CALIBRATION  — detects ChArUco board across multiple frames, averages for stability,
                    saves to disk. Does NOT require all 4 nodes to see the board at once.
  2. TRACKING     — uses YOLOv8n to detect people, converts foot-point to global 3D coords,
                    sends over OSC to Master.

Dependencies (see install_tracking.yml):
  - opencv-contrib-python >= 4.8
  - pyrealsense2
  - python-osc
  - ultralytics        (YOLOv8 — installs torch automatically)
  - numpy
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from ultralytics import YOLO
import socket
import threading
import time
import sys
import json
import os

# ---------------------------------------------------------------------------
# NETWORK SETUP
# ---------------------------------------------------------------------------
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

device_ip = get_ip().replace(".", "_")

if len(sys.argv) > 1:
    master_ip = sys.argv[1]
else:
    master_ip = "10.10.10.3"

OSC_PORT_OUT = 9001   # port Master listens on
OSC_PORT_IN  = 9003   # port this node listens on
OSC_ADDRESS  = "/person_" + device_ip

clients = [udp_client.SimpleUDPClient(master_ip, OSC_PORT_OUT)]

# ---------------------------------------------------------------------------
# REMOTE LOGGING
# ---------------------------------------------------------------------------
def remote_print(msg):
    print(msg)
    for client in clients:
        try:
            client.send_message("/log_" + device_ip, str(msg))
        except Exception:
            pass

def send_status(status: str):
    for client in clients:
        try:
            client.send_message("/status_" + device_ip, status)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# CALIBRATION CONFIG
# ---------------------------------------------------------------------------
SQUARES_X     = 5
SQUARES_Y     = 5
SQUARE_LENGTH = 0.22   # meters — match your printed board
MARKER_LENGTH = 0.165  # meters — match your printed board

# How many good frames to average before locking calibration.
# More = more stable, but takes longer to collect.
CALIB_FRAMES_NEEDED = 15

# Where to save/load calibration so a reboot doesn't require re-calibrating
CALIB_FILE = os.path.expanduser("~/supermesh_calib.json")

# ---------------------------------------------------------------------------
# YOLO CONFIG
# ---------------------------------------------------------------------------
# yolov8n.pt is the fastest nano model (~6MB). Downloads automatically on first run.
# Swap for yolov8s.pt if you want more accuracy and can spare the CPU.
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF  = 0.45   # confidence threshold — raise if you get false positives
PERSON_CLASS = 0    # COCO class 0 = person

# ---------------------------------------------------------------------------
# OSC COMMAND LISTENERS
# ---------------------------------------------------------------------------
force_calibrate = False
keep_running    = True

def calibrate_handler(address, *args):
    global force_calibrate
    remote_print("[OSC] CALIBRATE command received.")
    force_calibrate = True

def quit_handler(address, *args):
    remote_print("[OSC] QUIT command received. Forcing immediate shutdown...")
    os._exit(0)  # Hard kill to prevent pyrealsense2 from deadlocking on exit

dispatcher = Dispatcher()
dispatcher.map("/calibrate", calibrate_handler)
dispatcher.map("/quit",      quit_handler)
osc_server = BlockingOSCUDPServer(("0.0.0.0", OSC_PORT_IN), dispatcher)
threading.Thread(target=osc_server.serve_forever, daemon=True).start()

# ---------------------------------------------------------------------------
# CHARUCO BOARD
# ---------------------------------------------------------------------------
aruco_dict       = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board            = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board)

def find_floor_normal(depth_frame, intrinsics, depth_scale):
    """Sample depth points around the board area and fit a plane to find true floor normal."""
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    
    # Sample a grid of points from the lower half of the frame (floor area)
    points = []
    for y in range(h//2, h, 20):
        for x in range(w//4, 3*w//4, 20):
            d = depth_image[y, x] * depth_scale
            if 0.3 < d < 4.0:
                pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                points.append(pt)
    
    if len(points) < 20:
        return None
        
    points = np.array(points)
    
    # Fit a plane using SVD
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # normal to best-fit plane
    
    # Make sure normal points "up" (away from camera, toward ceiling)
    if normal[1] > 0:
        normal = -normal
        
    return normal, centroid

# ---------------------------------------------------------------------------
# REALSENSE PIPELINE
# ---------------------------------------------------------------------------
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
except Exception as e:
    send_status(f"ERROR: {e}")
    raise

depth_sensor  = profile.get_device().first_depth_sensor()
depth_scale   = depth_sensor.get_depth_scale()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics    = color_profile.get_intrinsics()

camera_matrix = np.array([[intrinsics.fx, 0,            intrinsics.ppx],
                           [0,            intrinsics.fy, intrinsics.ppy],
                           [0,            0,             1             ]])
dist_coeffs   = np.zeros(5)

align = rs.align(rs.stream.color)

# ---------------------------------------------------------------------------
# YOLO MODEL  (loads once, stays in memory)
# ---------------------------------------------------------------------------
remote_print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)
remote_print("YOLO model loaded.")

# ---------------------------------------------------------------------------
# CALIBRATION STATE
# ---------------------------------------------------------------------------
is_calibrated = False
global_R_inv  = None
global_tvec   = None

# Accumulated frames for averaging calibration
calib_rvecs = []
calib_tvecs = []

def save_calibration(R_inv, tvec):
    data = {
        "R_inv": R_inv.tolist(),
        "tvec":  tvec.tolist(),
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f)
    remote_print(f"Calibration saved to {CALIB_FILE}")

def load_calibration():
    """Returns (R_inv, tvec) if file exists, else None."""
    if not os.path.exists(CALIB_FILE):
        return None
    try:
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)
        R_inv = np.array(data["R_inv"])
        tvec  = np.array(data["tvec"])
        remote_print(f"Loaded saved calibration from {CALIB_FILE}")
        return R_inv, tvec
    except Exception as e:
        remote_print(f"Could not load calibration file: {e}")
        return None

def pixel_to_global(px, py, depth_m):
    """Convert a 2D pixel + metric depth into a global 3D coordinate."""
    cam_pt = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)
    cam_mat = np.array([[cam_pt[0]], [cam_pt[1]], [cam_pt[2]]])
    world   = global_R_inv @ (cam_mat - global_tvec)
    return float(world[0].item()), float(world[1].item()), float(world[2].item())

# ---------------------------------------------------------------------------
# TRY TO LOAD SAVED CALIBRATION
# ---------------------------------------------------------------------------
saved = load_calibration()
if saved:
    global_R_inv, global_tvec = saved
    is_calibrated = True
    send_status("Tracking")
    remote_print("=== RESUMED FROM SAVED CALIBRATION — TRACKING ===")
else:
    remote_print("=== HEADLESS SYSTEM READY ===")
    remote_print(f"Node IP: {device_ip}")
    remote_print("Show ChArUco board to camera, then send /calibrate from Master.")

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
last_status_time = 0.0

try:
    while keep_running:
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame    = aligned_frames.get_depth_frame()
        color_frame    = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        frame       = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        h_img, w_img = frame.shape[:2]

# ======================================================================
        # PHASE 1: CALIBRATION
        # Collects CALIB_FRAMES_NEEDED good detections then averages them.
        # Each node calibrates independently — no need to sync across cameras.
        # ======================================================================
        if not is_calibrated:
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(frame)

            board_visible = (charuco_corners is not None and len(charuco_corners) > 3)

            if board_visible:
                obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )

                if ok:
                    if force_calibrate or len(calib_rvecs) > 0:
                        if force_calibrate and len(calib_rvecs) == 0:
                            remote_print(f"Collecting {CALIB_FRAMES_NEEDED} calibration frames...")

                        # --- FLOOR NORMAL CONSTRAINT ---
                        # Sample depth points to find the true floor plane,
                        # then use that to correct the Y axis so it always
                        # points up regardless of camera tilt angle.
                        floor_points = []
                        for fy in range(depth_image.shape[0] // 2, depth_image.shape[0], 20):
                            for fx in range(depth_image.shape[1] // 4, 3 * depth_image.shape[1] // 4, 20):
                                d = depth_image[fy, fx] * depth_scale
                                if 0.3 < d < 4.0:
                                    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [fx, fy], d)
                                    floor_points.append(pt)

                        if len(floor_points) >= 20:
                            floor_pts_np = np.array(floor_points)
                            centroid = floor_pts_np.mean(axis=0)
                            centered = floor_pts_np - centroid
                            _, _, Vt_floor = np.linalg.svd(centered)
                            floor_normal = Vt_floor[-1]

                            # Ensure normal points away from floor toward ceiling.
                            # In camera space, Y increases downward, so the ceiling-
                            # facing normal should have a negative Y component.
                            if floor_normal[1] > 0:
                                floor_normal = -floor_normal

                            # Rebuild rotation matrix with true_y = floor normal
                            R_matrix, _ = cv2.Rodrigues(rvec)
                            true_y = floor_normal / np.linalg.norm(floor_normal)

                            # Recompute X and Z orthogonal to corrected Y
                            approx_z = R_matrix[:, 2]
                            true_x = np.cross(true_y, approx_z)
                            true_x = true_x / np.linalg.norm(true_x)
                            true_z = np.cross(true_x, true_y)
                            true_z = true_z / np.linalg.norm(true_z)

                            R_corrected = np.column_stack([true_x, true_y, true_z])
                            corrected_rvec = cv2.Rodrigues(R_corrected)[0]
                            calib_rvecs.append(corrected_rvec)
                        else:
                            # Not enough floor points — fall back to raw solvePnP result
                            remote_print("WARNING: Not enough floor points for floor constraint, using raw rvec")
                            calib_rvecs.append(rvec)

                        calib_tvecs.append(tvec)

                    # Enough frames collected — average and lock in
                    if len(calib_rvecs) >= CALIB_FRAMES_NEEDED:
                        R_matrices = [cv2.Rodrigues(r)[0] for r in calib_rvecs]
                        avg_R = np.mean(R_matrices, axis=0)
                        U, _, Vt = np.linalg.svd(avg_R)
                        avg_R = U @ Vt  # re-orthogonalize

                        global_R_inv = avg_R.T
                        global_tvec  = np.mean(calib_tvecs, axis=0)

                        is_calibrated = True
                        force_calibrate = False
                        calib_rvecs.clear()
                        calib_tvecs.clear()

                        save_calibration(global_R_inv, global_tvec)
                        remote_print(f"\n--- CALIBRATION LOCKED ({CALIB_FRAMES_NEEDED} frames averaged) ---")
                        send_status("Tracking")

                # Report board visible to master periodically
                now = time.time()
                if now - last_status_time > 2.0:
                    send_status("Board Visible")
                    progress = len(calib_rvecs)
                    if progress > 0:
                        remote_print(f"Calibrating... {progress}/{CALIB_FRAMES_NEEDED} frames")
                    last_status_time = now

            else:
                now = time.time()
                if now - last_status_time > 3.0:
                    send_status("No Board")
                    last_status_time = now

            continue  # Don't track until calibrated

        # ======================================================================
        # PHASE 2: YOLO PERSON TRACKING
        # Runs YOLOv8n on the color frame. For each detected person, samples
        # depth at the foot-point (bottom-center of bbox) for better ground-
        # plane accuracy than the torso center, then converts to global coords.
        # ======================================================================
        results = model(frame, classes=[PERSON_CLASS], conf=YOLO_CONF, verbose=False)

        person_count = 0
        for r in results:
            for i, box in enumerate(r.boxes):
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Foot-point: bottom-center of the bounding box.
                # This sits closer to the floor than the torso center,
                # giving a more accurate ground-plane X/Z position.
                foot_x = (x1 + x2) // 2
                foot_y = min(y2, h_img - 1)

                # Sample a small patch around the foot point for depth stability
                patch = depth_image[
                    max(0, foot_y - 10): min(h_img, foot_y + 10),
                    max(0, foot_x - 15): min(w_img, foot_x + 15)
                ]
                valid = patch[patch > 0]

                if len(valid) == 0:
                    continue

                # Use median (more robust to edge noise than mean)
                depth_m = float(np.median(valid)) * depth_scale

                # Sanity check — ignore implausible depths
                if not (0.3 < depth_m < 8.0):
                    continue

                gx, gy, gz = pixel_to_global(foot_x, foot_y, depth_m)

                if not (-10 < gx < 10 and -10 < gz < 10):
                    continue  # bad calibration artifact, discard

                for client in clients:
                    try:
                        client.send_message(
                            OSC_ADDRESS + f"/{person_count}",
                            [gx, gy, gz, conf]   # includes confidence score
                        )
                    except BlockingIOError:
                        pass

        # Heartbeat status so master knows node is alive
        now = time.time()
        if now - last_status_time > 5.0:
            send_status("Tracking")
            last_status_time = now

except KeyboardInterrupt:
    remote_print("Manually stopped.")
finally:
    remote_print("Stopping pipeline...")
    pipeline.stop()