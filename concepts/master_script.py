#Receives data from all 4 nodes, working
import time
import math
import numpy as np
import threading
import cv2
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# --- SECURITY & NETWORK ---
RECEIVE_IP = "0.0.0.0"
RECEIVE_PORT = 9001

# Only accept messages containing these strings
ALLOWED_BEE_IPS = ["10_10_10_19", "10_10_10_20", "10_10_10_21", "10_10_10_22"]

# Clients to talk BACK to the Beelinks (for remote calibration)
BEELINK_PORT = 9003
bee_clients = [SimpleUDPClient(ip.replace("_", "."), BEELINK_PORT) for ip in ALLOWED_BEE_IPS]

# Audio Software Output
AUDIO_SOFTWARE_IP = "127.0.0.1"
AUDIO_SOFTWARE_PORT = 9002
audio_client = SimpleUDPClient(AUDIO_SOFTWARE_IP, AUDIO_SOFTWARE_PORT)

# --- ALGORITHM GLOBALS ---
CLUSTER_RADIUS = 0.6
SMOOTHING_FACTOR = 0.3
STALE_DATA_TIMEOUT = 0.2

raw_camera_points = {}
active_people = []
next_person_id = 1

class TrackedPerson:
    def __init__(self, person_id, x, y, z):
        self.id = person_id
        self.x, self.y, self.z = x, y, z
        self.missed_frames = 0

# --- UI CONTROLS ---
def update_radius(val):
    global CLUSTER_RADIUS
    CLUSTER_RADIUS = val / 100.0  # Convert cm slider to meters

def update_smoothing(val):
    global SMOOTHING_FACTOR
    SMOOTHING_FACTOR = val / 100.0

cv2.namedWindow("MESH God View", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Cluster Radius (cm)", "MESH God View", int(CLUSTER_RADIUS * 100), 150, update_radius)
cv2.createTrackbar("Smoothing (%)", "MESH God View", int(SMOOTHING_FACTOR * 100), 100, update_smoothing)

# --- OSC RECEIVER ---
def osc_handler(address, *args):
    # SECURITY GATE: Drop message if not from an allowed IP
    if not any(ip in address for ip in ALLOWED_BEE_IPS):
        return
        
    if "blob" in address and len(args) == 3:
        raw_camera_points[address] = [args[0], args[1], args[2], time.time()]

# --- MAPPING MATH ---
def meters_to_pixels(x, z, map_size=800, scale=4.0):
    # Maps a 3D coordinate (in meters) to a 2D pixel coordinate for the visualizer.
    # Assumes the ChArUco board (0,0) is in the exact center of the map.
    px = int((x / scale) * (map_size / 2) + (map_size / 2))
    pz = int((z / scale) * (map_size / 2) + (map_size / 2))
    return px, pz

# --- MAIN PROCESS ---
def start_master():
    global active_people, next_person_id
    
    # Start the OSC Receiver in the background
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(osc_handler)
    server = BlockingOSCUDPServer((RECEIVE_IP, RECEIVE_PORT), dispatcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    
    print("Master Node Active. Listening for approved IPs.")
    
    while True:
        current_time = time.time()
        
        # 1. Clean Stale Data
        valid_points = []
        keys_to_delete = [k for k, v in raw_camera_points.items() if current_time - v[3] > STALE_DATA_TIMEOUT]
        for k in keys_to_delete: del raw_camera_points[k]
        for v in raw_camera_points.values(): valid_points.append(np.array([v[0], v[1], v[2]]))

        # 2. Clustering
        clusters = []
        for pt in valid_points:
            matched = False
            for i, center in enumerate(clusters):
                if np.linalg.norm(pt - center) < CLUSTER_RADIUS:
                    clusters[i] = (center + pt) / 2.0
                    matched = True
                    break
            if not matched: clusters.append(pt)

        # 3. ID Matching & Smoothing
        updated_people = []
        for cluster in clusters:
            best_match, best_dist = None, 1.5
            for person in active_people:
                dist = np.linalg.norm(cluster - np.array([person.x, person.y, person.z]))
                if dist < best_dist:
                    best_match, best_dist = person, dist
            
            if best_match:
                best_match.x = (best_match.x * (1 - SMOOTHING_FACTOR)) + (cluster[0] * SMOOTHING_FACTOR)
                best_match.z = (best_match.z * (1 - SMOOTHING_FACTOR)) + (cluster[2] * SMOOTHING_FACTOR)
                best_match.missed_frames = 0
                updated_people.append(best_match)
                active_people.remove(best_match)
            else:
                updated_people.append(TrackedPerson(next_person_id, cluster[0], cluster[1], cluster[2]))
                next_person_id += 1

        for missing in active_people:
            missing.missed_frames += 1
            if missing.missed_frames < 5: updated_people.append(missing)

        active_people = updated_people

        # --- DRAW THE GUI MAP ---
        map_size = 800
        canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # Draw a grid and center point
        cv2.line(canvas, (0, map_size//2), (map_size, map_size//2), (50, 50, 50), 1)
        cv2.line(canvas, (map_size//2, 0), (map_size//2, map_size), (50, 50, 50), 1)
        cv2.circle(canvas, (map_size//2, map_size//2), 5, (255, 255, 255), -1)
        cv2.putText(canvas, "Origin (ChArUco)", (map_size//2 + 10, map_size//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw Raw Camera Points (Tiny red dots)
        for pt in valid_points:
            px, pz = meters_to_pixels(pt[0], pt[2], map_size)
            cv2.circle(canvas, (px, pz), 4, (0, 0, 255), -1)

        # Draw Fused People (Large green circles)
        for p in active_people:
            px, pz = meters_to_pixels(p.x, p.z, map_size)
            # Draw the cluster radius boundary to visually debug!
            radius_px = int((CLUSTER_RADIUS / 4.0) * (map_size / 2)) 
            cv2.circle(canvas, (px, pz), radius_px, (0, 50, 0), 1) 
            
            cv2.circle(canvas, (px, pz), 12, (0, 255, 0), -1)
            cv2.putText(canvas, f"ID:{p.id}", (px + 15, pz + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Send OSC
            audio_client.send_message(f"/person/{p.id}", [float(p.x), float(p.y), float(p.z)])

        cv2.putText(canvas, "PRESS 'C' TO CALIBRATE ALL CAMERAS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("MESH God View", canvas)

        # 4. REMOTE COMMANDS
        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Blasting Calibration Command to Beelinks!")
            for client in bee_clients:
                client.send_message("/calibrate", 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_master()