import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from pythonosc import udp_client
import socket
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

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
OSC_IPS = ["10.10.10.25"]
OSC_PORT = 9001
OSC_ADDRESS = "/blob" + device_ip
clients = [udp_client.SimpleUDPClient(ip, OSC_PORT) for ip in OSC_IPS]

# --- REMOTE CALIBRATION LISTENER ---
def remote_calibrate_handler(address, *args):
    global force_calibrate
    print("Received remote calibration command from Master!")
    force_calibrate = True

force_calibrate = False
dispatcher = Dispatcher()
dispatcher.set_default_handler(remote_calibrate_handler)
# We use port 9003 to receive, so it doesn't conflict with the sending port
server = BlockingOSCUDPServer(("0.0.0.0", 9003), dispatcher) 
import threading
threading.Thread(target=server.serve_forever, daemon=True).start()

# --- CHARUCO BOARD SETUP ---
# Adjust these measurements (in meters) to match your physical printed board!
SQUARES_X = 3
SQUARES_Y = 3
SQUARE_LENGTH = 0.027 
MARKER_LENGTH = 0.02 

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# The modern OpenCV 4.8+ way to initialize the detector
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

# Get depth scale and camera intrinsics (Crucial for 3D math)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# We need the camera matrix for OpenCV pose estimation
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
dist_coeffs = np.zeros(5) # RealSense handles its own distortion

align_to = rs.stream.color
align = rs.align(align_to)

# --- BLOB SETUP ---
backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)
MIN_BLOB_AREA = 2500
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

cv2.namedWindow("Tracking Feed", cv2.WINDOW_NORMAL)

print("System Ready. Place board in center of room.")
print("Press 'c' when board is detected to Calibrate.")
print("Press 'q' to exit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        h_img, w_img = frame.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # ==========================================
        # PHASE 1: CALIBRATION MODE
        # ==========================================
        if not is_calibrated:
            # The New OpenCV 4.8+ Detection Method
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)
            
            # If it sees enough of the board to do 3D math
            if charuco_corners is not None and len(charuco_corners) > 3:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (255, 0, 0))
                
                # The New OpenCV 4.8+ Pose Estimation Method
                obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                
                if success:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.2)
                    cv2.putText(frame, "BOARD DETECTED! PRESS 'C' TO LOCK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # If user presses 'c', lock in the math!
                    if key == ord('c') or force_calibrate:
                        R_matrix, _ = cv2.Rodrigues(rvec)
                        global_R_inv = R_matrix.T
                        global_tvec = tvec
                        is_calibrated = True
                        print("\n--- CALIBRATION SUCCESSFUL ---")
                        print("Switched to Global Tracking Mode.")
            else:
                cv2.putText(frame, "Looking for board...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Tracking Feed", frame)
            continue # Skip blob tracking until calibratedq


        # ==========================================
        # PHASE 2: GLOBAL TRACKING MODE
        # ==========================================
        # 1. Motion mask
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

                # Get Average Depth
                patch = depth_image[max(0, center_y-5):min(h_img, center_y+5), 
                                    max(0, center_x-5):min(w_img, center_x+5)]
                valid_depths = patch[patch > 0]
                
                if len(valid_depths) > 0:
                    avg_depth = np.mean(valid_depths) * depth_scale
                    
                    # --- THE 3D MAGIC ---
                    # 1. Ask RealSense to turn the 2D pixel + Depth into a 3D Camera Coordinate
                    camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, [center_x, center_y], avg_depth)
                    
                    # Convert to numpy array for matrix math: shape (3, 1)
                    cam_pt_matrix = np.array([[camera_point[0]], [camera_point[1]], [camera_point[2]]])
                    
                    # 2. Translate to Global Space using our ChArUco matrix!
                    # Formula: World_Point = R_inverse * (Camera_Point - Translation_Vector)
                    global_pt_matrix = global_R_inv @ (cam_pt_matrix - global_tvec)
                    
                    global_X = float(global_pt_matrix[0][0])
                    global_Y = float(global_pt_matrix[1][0])
                    global_Z = float(global_pt_matrix[2][0])

                    # Print Global Coordinates
                    print(f"Blob {blob_count} | Global X: {global_X:.2f}m, Y: {global_Y:.2f}m, Z: {global_Z:.2f}m")

                    # Send unified OSC packet
                    for client in clients:
                        try:
                            client.send_message(OSC_ADDRESS + f"/{blob_count}", [global_X, global_Y, global_Z])
                        except BlockingIOError:
                            pass
                
                # Draw visuals
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        cv2.putText(frame, "TRACKING (Global Space)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Tracking Feed", frame)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()