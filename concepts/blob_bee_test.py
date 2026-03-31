import cv2
import numpy as np
import pyrealsense2 as rs
from pythonosc import udp_client
import socket
from screeninfo import get_monitors

# Detect primary monitor resolution
monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height
new_w = screen_w
new_h = screen_h

# Get device IP for OSC routing
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

# OSC SETTINGS
OSC_IPS = ["10.10.10.19", "10.10.10.20", "10.10.10.21"]
OSC_PORT = 9001
OSC_ADDRESS = "/blob" + device_ip
clients = [udp_client.SimpleUDPClient(ip, OSC_PORT) for ip in OSC_IPS]

# --- REALSENSE CAPTURE SETUP ---
pipeline = rs.pipeline()
config = rs.config()

# Enable BOTH color and depth streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get the depth sensor's depth scale (used to convert depth units to actual meters)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an alignment object. We want to warp the depth map to align with the color video.
align_to = rs.stream.color
align = rs.align(align_to)

# --- BACKGROUND & BLOB MODEL ---
backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)
MIN_BLOB_AREA = 2500  # Increased slightly to ignore smaller room noise

# Create a "thick brush" for fusing fragmented blobs together
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# Fullscreen window
cv2.namedWindow("Blob Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Blob Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Blob Tracking + Depth started. Press 'q' to exit.")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Flip BOTH frames so they act like a mirror and the coordinates still match
        frame = cv2.flip(frame, 1)
        depth_image = cv2.flip(depth_image, 1)

        # 1. Motion mask generation
        mask = backsub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        # 2. Cleanup and Fusion (This fixes the "multiple blobs per person" issue)
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.dilate(mask, dilation_kernel, iterations=4) # Heavily smear the blobs together

        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_count = 0
        h_img, w_img = frame.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > MIN_BLOB_AREA:
                blob_count += 1
                
                # Get bounding box and center
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + (w // 2)
                center_y = y + (h // 2)

                # --- DEPTH CALCULATION ---
                # Grab a 10x10 pixel square around the center point
                # The max/min functions ensure we don't accidentally try to sample off the edge of the screen
                patch = depth_image[max(0, center_y-5) : min(h_img, center_y+5), 
                                    max(0, center_x-5) : min(w_img, center_x+5)]
                
                # Filter out zeroes (dead depth pixels)
                valid_depths = patch[patch > 0]
                
                if len(valid_depths) > 0:
                    # Average the valid pixels and multiply by the scale to get actual meters
                    avg_depth_meters = np.mean(valid_depths) * depth_scale
                else:
                    avg_depth_meters = 0.0

                # Print to terminal
                print(f"Blob {blob_count} | Center: ({center_x}, {center_y}) | Depth: {avg_depth_meters:.2f}m")

                # Send OSC
                for client in clients:
                    try:
                        # Bundle X, Y, and Z into a single message
                        client.send_message(OSC_ADDRESS + f"/{blob_count}", 
                                            [float(center_x), float(center_y), float(avg_depth_meters)])
                    except BlockingIOError:
                        # If the OS network buffer is full, just silently drop the packet 
                        # instead of crashing the whole program.
                        pass

                # --- DRAWING VISUALS ---
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Blob {blob_count} ({avg_depth_meters:.2f}m)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display formatting
        img_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        x_offset = (screen_w - new_w) // 2
        y_offset = (screen_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        cv2.imshow("Blob Tracking", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()