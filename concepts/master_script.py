import time
import numpy as np
import threading
import cv2
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# --- NETWORK ---
RECEIVE_IP   = "0.0.0.0"
RECEIVE_PORT = 9001

ALLOWED_BEE_IPS = ["10_10_10_19", "10_10_10_20", "10_10_10_21", "10_10_10_22"]

BEELINK_PORT = 9003
bee_clients  = [SimpleUDPClient(ip.replace("_", "."), BEELINK_PORT) for ip in ALLOWED_BEE_IPS]

AUDIO_SOFTWARE_IP   = "127.0.0.1"
AUDIO_SOFTWARE_PORT = 9002
audio_client = SimpleUDPClient(AUDIO_SOFTWARE_IP, AUDIO_SOFTWARE_PORT)

# --- ALGORITHM GLOBALS ---
CLUSTER_RADIUS    = 0.6
SMOOTHING_FACTOR  = 0.3
STALE_DATA_TIMEOUT = 0.2

raw_camera_points = {}
active_people     = []
next_person_id    = 1

camera_statuses = {ip: "Offline / No Board" for ip in ALLOWED_BEE_IPS}

class TrackedPerson:
    def __init__(self, person_id, x, y, z):
        self.id = person_id
        self.x, self.y, self.z = x, y, z
        self.missed_frames = 0

cv2.namedWindow("MESH Minimal View", cv2.WINDOW_NORMAL)

# --- OSC RECEIVER ---
def osc_handler(address, *args):
    node_ip = next((ip for ip in ALLOWED_BEE_IPS if ip in address), None)
    if not node_ip:
        return

    # NEW: headless_script sends /person_<ip>/<n> with [x, y, z, conf]
    # OLD: sent /blob_<ip>/<n> with [x, y, z]
    # Handle both so you don't break anything mid-transition.
    if ("person" in address or "blob" in address) and len(args) >= 3:
        raw_camera_points[address] = [args[0], args[1], args[2], time.time()]
        camera_statuses[node_ip] = "Tracking"

    elif "status" in address and len(args) == 1:
        if camera_statuses[node_ip] != args[0]:
            print(f"[STATUS] Node {node_ip}: {args[0]}")
        camera_statuses[node_ip] = args[0]

    elif "log" in address and len(args) == 1:
        print(f"[{node_ip}] {args[0]}")

# --- MAPPING MATH ---
def meters_to_pixels(x, z, map_size=800, scale=4.0):
    px = int((x / scale) * (map_size / 2) + (map_size / 2))
    pz = int((z / scale) * (map_size / 2) + (map_size / 2))
    return px, pz

# --- MAIN LOOP ---
def start_master():
    global active_people, next_person_id

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(osc_handler)
    server = BlockingOSCUDPServer((RECEIVE_IP, RECEIVE_PORT), dispatcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("Master Node Active.")

    while True:
        current_time = time.time()

        # 1. Clean stale data
        keys_to_delete = [k for k, v in raw_camera_points.items()
                          if current_time - v[3] > STALE_DATA_TIMEOUT]
        for k in keys_to_delete:
            del raw_camera_points[k]
        valid_points = [np.array([v[0], v[1], v[2]]) for v in raw_camera_points.values()]

        # 2. Clustering
        clusters = []
        for pt in valid_points:
            matched = False
            for i, center in enumerate(clusters):
                if np.linalg.norm(pt - center) < CLUSTER_RADIUS:
                    clusters[i] = (center + pt) / 2.0
                    matched = True
                    break
            if not matched:
                clusters.append(pt)

        # 3. ID matching + smoothing
        updated_people = []
        for cluster in clusters:
            best_match, best_dist = None, 1.5
            for person in active_people:
                dist = np.linalg.norm(cluster - np.array([person.x, person.y, person.z]))
                if dist < best_dist:
                    best_match, best_dist = person, dist

            if best_match:
                best_match.x += SMOOTHING_FACTOR * (cluster[0] - best_match.x)
                best_match.z += SMOOTHING_FACTOR * (cluster[2] - best_match.z)
                best_match.missed_frames = 0
                updated_people.append(best_match)
                active_people.remove(best_match)
            else:
                updated_people.append(TrackedPerson(next_person_id, *cluster))
                next_person_id += 1

        for missing in active_people:
            missing.missed_frames += 1
            if missing.missed_frames < 5:
                updated_people.append(missing)

        active_people = updated_people

        # --- UI ---
        map_size = 800
        canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)

        # People dots
        for p in active_people:
            px, pz = meters_to_pixels(p.x, p.z, map_size)
            cv2.circle(canvas, (px, pz), 15, (0, 255, 0), -1)
            cv2.putText(canvas, f"ID:{p.id}", (px + 20, pz + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            audio_client.send_message(f"/person/{p.id}", [float(p.x), float(p.y), float(p.z)])

        # Node status panel
        cv2.putText(canvas, "NODE STATUS:", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset = 60
        for ip, status in camera_statuses.items():
            if status == "Tracking":
                color = (0, 255, 0)
            elif status == "Board Visible":
                color = (0, 255, 255)
            elif "Calibrating" in status:
                color = (0, 165, 255)   # orange while collecting frames
            else:
                color = (100, 100, 100)

            cv2.putText(canvas, f"Node {ip[-2:]}: {status}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        cv2.putText(canvas, "'C' = Calibrate | 'Q' = Quit", (20, map_size - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("MESH Minimal View", canvas)

        # 4. Key commands
        key = cv2.waitKey(33) & 0xFF

        if key == ord('q'):
            print("Sending QUIT to all Beelinks...")
            for client in bee_clients:
                try: client.send_message("/quit", 1)
                except Exception: pass
            time.sleep(0.1)
            break

        elif key == ord('c'):
            print("Sending CALIBRATE to all Beelinks...")
            for client in bee_clients:
                try: client.send_message("/calibrate", 1)
                except Exception: pass

    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_master()