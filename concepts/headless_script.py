import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from pythonosc import udp_client
import socket
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading
import time

# --- NETWORK SETUP ---
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
OSC_IPS = ["10.10.10.3"]
OSC_PORT = 9001
OSC_ADDRESS = "/blob_" + device_ip
clients = [udp_client.SimpleUDPClient(ip, OSC_PORT) for ip in OSC_IPS]

# --- REMOTE COMMAND LISTENERS ---
force_calibrate = False
keep_running = True # New flag to control the main loop

def calibrate_handler(address, *args):
    global force_calibrate
    print("\n[OSC] Received CALIBRATE command from Master!")
    force_calibrate = True

def quit_handler(address, *args):
    global keep_running
    print("\n[OSC] Received QUIT command from Master! Initiating shutdown...")
    keep_running = False

dispatcher = Dispatcher()
# Map specific addresses instead of a catch-all default
dispatcher.map("/calibrate", calibrate_handler)
dispatcher.map("/quit", quit_handler)

server = BlockingOSCUDPServer(("0.0.0.0", 9003), dispatcher)
threading.Thread(target=server.serve_forever, daemon=True).start()

# --- CHARUCO BOARD SETUP ---
SQUARES_X = 5
SQUARES_Y = 5
SQUARE_LENGTH = 0.22
MARKER_LENGTH = 0.165

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
charuco_detector = cv2.aruco.CharucoDetector(board)

# Variables to hold our 3D transformation data once calibrated
is_calibrated = False
global_R_inv = None
global_tvec = None

# --- REALSENSE SETUP ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.zeros(5) 

align_to = rs.stream.color
align = rs.align(align_to)

# --- BLOB SETUP ---
backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)
MIN_BLOB_AREA = 2500
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

print("=== HEADLESS SYSTEM READY ===")
print(f"Node IP: {device_ip}")
print("Waiting for Master Node to send Calibration Command...")

last_print_time = time.time()

try:
    while keep_running:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        h_img, w_img = frame.shape[:2]

        # ==========================================
        # PHASE 1: CALIBRATION MODE
        # ==========================================
        if not is_calibrated:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
            
            if charuco_corners is not None and len(charuco_corners) > 3:
                obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                if success:
                    # Throttle the "Board Detected" print & OSC message so it doesn't spam
                    if time.time() - last_print_time > 2.0:
                        print("Board visible. Ready and waiting for remote calibration trigger...")
                        last_print_time = time.time()
                        
                        # --- ADD THIS: Tell master the board is visible ---
                        for client in clients:
                            try:
                                client.send_message("/status_" + device_ip, "Board Visible")
                            except BlockingIOError:
                                pass
                    
                    if force_calibrate:
                        R_matrix, _ = cv2.Rodrigues(rvec)
                        global_R_inv = R_matrix.T
                        global_tvec = tvec
                        is_calibrated = True
                        force_calibrate = False 
                        print("\n--- CALIBRATION SUCCESSFUL ---")
                        
                        # --- ADD THIS: Tell master we are calibrated ---
                        for client in clients:
                            try:
                                client.send_message("/status_" + device_ip, "Tracking")
                            except BlockingIOError:
                                pass
            continue


        # ==========================================
        # PHASE 2: GLOBAL TRACKING MODE
        # ==========================================
        mask = backsub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.dilate(mask, dilation_kernel, iterations=4)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_BLOB_AREA:
                blob_count += 1
                
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                patch = depth_image[max(0, center_y-5):min(h_img, center_y+5), 
                                    max(0, center_x-5):min(w_img, center_x+5)]
                valid_depths = patch[patch > 0]
                
                if len(valid_depths) > 0:
                    avg_depth = np.mean(valid_depths) * depth_scale
                    
                    camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], avg_depth)
                    cam_pt_matrix = np.array([[camera_point[0]], [camera_point[1]], [camera_point[2]]])
                    
                    global_pt_matrix = global_R_inv @ (cam_pt_matrix - global_tvec)
                    
                    global_X = float(global_pt_matrix[0][0])
                    global_Y = float(global_pt_matrix[1][0])
                    global_Z = float(global_pt_matrix[2][0])

                    # Optional: Comment this out later if you don't want massive log files
                    # print(f"Blob {blob_count} | X: {global_X:.2f}m, Y: {global_Y:.2f}m, Z: {global_Z:.2f}m")

                    for client in clients:
                        try:
                            client.send_message(OSC_ADDRESS + f"/{blob_count}", [global_X, global_Y, global_Z])
                        except BlockingIOError:
                            pass

except KeyboardInterrupt:
    print("\nProcess manually terminated via KeyboardInterrupt.")

finally:
    print("Stopping pipeline...")
    pipeline.stop()
