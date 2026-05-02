# --- MASTER ENHANCED MONITOR ---
# This is the most advanced version of the SuperMesh Master script.
# Features: YOLOv8 Person Detection (on nodes), Multi-Camera Fusion, Motion Trails, and Grid Overlay.
# Controls: Arrows to Pan, +/- to Zoom, 'O' to Re-center on detected persons.
# Note: This is significantly more accurate than the legacy blob-detection version.

import time
import math
import numpy as np
import threading
import cv2
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from collections import deque

# --- CONFIGURATION ---
RECEIVE_IP   = "0.0.0.0"
RECEIVE_PORT = 9001

ALLOWED_BEE_IPS = ["10_10_10_19", "10_10_10_20", "10_10_10_21", "10_10_10_22"]
NODE_COLORS = {
    "10_10_10_19": (0, 0, 255),   # Red
    "10_10_10_20": (255, 0, 0),   # Blue
    "10_10_10_21": (0, 255, 255), # Yellow
    "10_10_10_22": (255, 0, 255)  # Magenta
}

PD_IP   = "127.0.0.1"  # change if Pd is on a different machine
PD_PORT = 9005

pd_client = SimpleUDPClient(PD_IP, PD_PORT)

BEELINK_PORT = 9003
bee_clients  = [SimpleUDPClient(ip.replace("_", "."), BEELINK_PORT) for ip in ALLOWED_BEE_IPS]

# --- ALGORITHM TUNING (Now adjustable via keyboard) ---
CLUSTER_RADIUS    = 1.1  # meters - how close points must be to merge
SMOOTHING_FACTOR  = 0.15 # lower = smoother
STALE_DATA_TIMEOUT = 0.4
MAX_TRAIL_POINTS   = 30
GHOST_TIMEOUT     = 0.8

# --- VIEW SETTINGS ---
MAP_SIZE = 900
scale = 8.0     # meters visible
offset_x = 0.0  # panning
offset_z = 0.0

raw_camera_points = {} # stores [x, y, z, time, node_ip]
active_people     = []
next_person_id    = 1
camera_statuses   = {ip: "Offline" for ip in ALLOWED_BEE_IPS}

class TrackedPerson:
    def __init__(self, person_id, x, y, z):
        self.id = person_id
        self.x, self.y, self.z = x, y, z
        self.history = deque(maxlen=MAX_TRAIL_POINTS)
        self.last_seen = time.time()
        self.missed_frames = 0

    def update(self, nx, ny, nz):
        self.history.append((self.x, self.z))
        self.x += SMOOTHING_FACTOR * (nx - self.x)
        self.y += SMOOTHING_FACTOR * (ny - self.y)
        self.z += SMOOTHING_FACTOR * (nz - self.z)
        self.last_seen = time.time()
        self.missed_frames = 0

# --- OSC RECEIVER ---
# --- OSC RECEIVER ---
def osc_handler(address, *args):
    node_ip = next((ip for ip in ALLOWED_BEE_IPS if ip in address), None)
    if not node_ip: return

    if ("person" in address or "blob" in address) and len(args) >= 3:
        x, y, z = args[0], args[1], args[2]
        
        # --- NEW: PROTECT AGAINST NaN VALUES ---
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            return  # Silently drop this corrupt packet
            
        raw_camera_points[address] = [x, y, z, time.time(), node_ip]
        camera_statuses[node_ip] = "Tracking"
    elif "status" in address and len(args) == 1:
        camera_statuses[node_ip] = args[0]

# --- MAPPING MATH ---
def meters_to_pixels(x, z):
    # Apply pan and scale
    px = int(((x + offset_x) / scale) * (MAP_SIZE) + (MAP_SIZE / 2))
    pz = int(((z + offset_z) / scale) * (MAP_SIZE) + (MAP_SIZE / 2))
    return px, pz

def start_master():
    global active_people, next_person_id, scale, offset_x, offset_z, CLUSTER_RADIUS
    
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(osc_handler)
    server = BlockingOSCUDPServer((RECEIVE_IP, RECEIVE_PORT), dispatcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cv2.namedWindow("SuperMesh Enhanced Monitor", cv2.WINDOW_NORMAL)
    print("Enhanced Master Node Active. Controls: +/- Zoom, Arrows Pan, 'O' Re-center on detected person.")

    while True:
        current_time = time.time()
        
        # 1. Clean Stale Raw Data
        keys_to_delete = [k for k, v in raw_camera_points.items() if current_time - v[3] > STALE_DATA_TIMEOUT]
        for k in keys_to_delete: del raw_camera_points[k]
        
        # 2. Strong Clustering (Merge multiple camera views into single person clusters)
        clusters = [] 
        for addr, p in list(raw_camera_points.items()):
            matched = False
            pt = np.array([p[0], p[1], p[2]])
            for cluster in clusters:
                center = np.array([cluster[0], cluster[1], cluster[2]])
                if np.linalg.norm(np.array([pt[0]-center[0], pt[2]-center[2]])) < CLUSTER_RADIUS:
                    # Rolling average for the cluster center
                    count = cluster[4]
                    cluster[0] = (cluster[0] * count + p[0]) / (count + 1)
                    cluster[1] = (cluster[1] * count + p[1]) / (count + 1)
                    cluster[2] = (cluster[2] * count + p[2]) / (count + 1)
                    cluster[3].append(p[4]) # nodes contributing
                    cluster[4] += 1        # total points contributing
                    matched = True
                    break
            if not matched:
                clusters.append([p[0], p[1], p[2], [p[4]], 1])

        # 3. ID Matching (Link clusters to persistent TrackedPerson objects)
        new_active_list = []
        for c in clusters:
            best_match, best_dist = None, 1.8 # allow 1.8m jump max
            c_pt = np.array([c[0], c[1], c[2]])
            for person in active_people:
                dist = math.sqrt((c_pt[0]-person.x)**2 + (c_pt[2]-person.z)**2)
                if dist < best_dist:
                    best_match, best_dist = person, dist
            
            if best_match:
                best_match.update(c[0], c[1], c[2])
                new_active_list.append(best_match)
                active_people.remove(best_match)
            else:
                new_active_list.append(TrackedPerson(next_person_id, c[0], c[1], c[2]))
                next_person_id += 1

        # Ghosting logic
        for p in active_people:
            if current_time - p.last_seen < GHOST_TIMEOUT:
                p.missed_frames += 1
                new_active_list.append(p)
        active_people = new_active_list

        for p in active_people:
            if p.missed_frames == 0:  # only send confirmed detections, not ghosts
                pd_client.send_message(f"/person/{p.id}", [p.x, p.z])
            
        # --- DRAWING ---
        canvas = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        
        # Grid (1 meter spacing)
        for i in range(-10, 11):
            x1, z1 = meters_to_pixels(i, -10)
            x2, z2 = meters_to_pixels(i, 10)
            cv2.line(canvas, (x1, z1), (x2, z2), (20, 20, 20), 1)
            x1, z1 = meters_to_pixels(-10, i)
            x2, z2 = meters_to_pixels(10, i)
            cv2.line(canvas, (x1, z1), (x2, z2), (20, 20, 20), 1)

        # Draw RAW camera points (dots)
        for addr, p in list(raw_camera_points.items()):
            rx, rz = meters_to_pixels(p[0], p[2])
            color = NODE_COLORS.get(p[4], (100, 100, 100))
            cv2.circle(canvas, (rx, rz), 4, color, -1)

        # Draw Fused People
        for p in active_people:
            px, pz = meters_to_pixels(p.x, p.z)
            
            # Trails
            if len(p.history) > 1:
                pts = [meters_to_pixels(hx, hz) for hx, hz in p.history]
                for i in range(len(pts)-1):
                    alpha = (i / len(pts))
                    c = (int(0 * alpha), int(255 * alpha), int(0 * alpha))
                    cv2.line(canvas, pts[i], pts[i+1], c, 3)

            # Dot & Label
            dot_color = (0, 255, 0) if p.missed_frames == 0 else (0, 60, 0)
            cv2.circle(canvas, (px, pz), 15, dot_color, -1)
            cv2.putText(canvas, f"ID:{p.id}", (px + 20, pz - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, f"Z: {p.z:.1f}m", (px + 20, pz + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # UI Overlay
        cv2.putText(canvas, "SUPERMESH FLEET MONITOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_off = 80
        for ip in ALLOWED_BEE_IPS:
            status = camera_statuses[ip]
            color = NODE_COLORS.get(ip, (255, 255, 255))
            cv2.rectangle(canvas, (20, y_off-18), (38, y_off), color, -1)
            cv2.putText(canvas, f"Node {ip[-2:]}: {status}", (50, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y_off += 30

        cv2.putText(canvas, f"Zoom: {scale:.1f}m | Pan: {offset_x:.1f}, {offset_z:.1f}", (20, MAP_SIZE-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(canvas, "Arrows=Pan | +/-=Zoom | O=Center | Q=Quit", (20, MAP_SIZE-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow("SuperMesh Enhanced Monitor", canvas)

        # KEY COMMANDS
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): 
            print("Blasting QUIT command to Beelink nodes...")
            for client in bee_clients: 
                client.send_message("/quit", 1)  # Sends a shutdown signal to the nodes
            break
        elif key == ord('='): scale = max(1.0, scale - 0.5) # Zoom in
        elif key == ord('-'): scale += 0.5                  # Zoom out
        elif key == ord('0'): offset_x = offset_z = 0.0; scale = 8.0 # Reset
        elif key == 81: offset_x += 0.2 # Left arrow
        elif key == 83: offset_x -= 0.2 # Right arrow
        elif key == 82: offset_z += 0.2 # Up arrow
        elif key == 84: offset_z -= 0.2 # Down arrow
        elif key == ord('o') and len(active_people) > 0:
            # Re-center on the first person found
            offset_x = -active_people[0].x
            offset_z = -active_people[0].z
        elif key == ord('c'):
            print("Blasting CALIBRATE...")
            for client in bee_clients: client.send_message("/calibrate", 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_master()
