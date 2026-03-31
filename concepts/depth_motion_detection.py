import cv2
import numpy as np
import pyrealsense2 as rs
from pythonosc import udp_client
import math
import socket
from screeninfo import get_monitors  # <-- screeninfo only

# Detect primary monitor resolution
monitor = get_monitors()[0]
screen_w, screen_h = monitor.width, monitor.height
new_w = screen_w
new_h = screen_h

# use this later
max_angle = 0


def map_range(value, input_start, input_end, output_start, output_end):
    """
    Maps a value from one range to another.
    """
    # Calculate the ratio of the value's position within the input range (0 to 1)
    input_span = input_end - input_start
    output_span = output_end - output_start

    if input_span == 0:
        # Avoid division by zero if the input range is a single point
        return output_start

    value_scaled = float(value - input_start) / float(input_span)

    # Convert the 0-1 range into a value in the output range
    return output_start + (value_scaled * output_span)


# Get device IP
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
OSC_ADDRESS = "/motion" + device_ip
clients = [udp_client.SimpleUDPClient(ip, OSC_PORT) for ip in OSC_IPS]

# REALSENSE CAPTURE SETUP
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream color (matching standard webcam output)
# Adjust resolution and framerate here if needed (e.g., 640x480 at 30fps)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# BACKGROUND MODEL
backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)

prev_gray = None
step = 8  # optical flow sampling step

# Fullscreen window
cv2.namedWindow("Motion Mask", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Motion Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Motion entropy + vector field. Press 'q' to exit.")

try:
    while True:
        # Wait for a coherent pair of frames from the RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        total_pixels = w * h

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue

        # Motion mask
        mask = backsub.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        # mask = cv2.medianBlur(mask, 11)
        # mask = cv2.dilate(mask, None, iterations=2)
        motion_pixels = np.count_nonzero(mask)

        # Entropy
        p = motion_pixels / float(total_pixels)
        entropy = -p * math.log(p) * 2 if p > 0 else 0.0

        # Send OSC
        for client in clients:
            client.send_message(OSC_ADDRESS, float(entropy))

        # Optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray

        # Create visualization
        img = np.zeros_like(frame)
        cv2.putText(img,
                    f"entropy: {entropy:.4f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2)

        degree_sum = 0
        count = 1
        
        # Draw optical flow arrows
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x] < 128:
                    continue
                fx, fy = flow[y, x]

                end_x = int(x + fx * 5)
                end_y = int(y + fy * 5)
                radians = math.atan2(fy, fx)
                degrees = abs(math.degrees(radians))

                # sum of each angle
                degree_sum = degree_sum + degrees
                count = count + 1

                absolute = math.sqrt(fx ** 2 + fy ** 2)

                mapped_val = map_range(int(degrees), 0, 360, 0, 255)
                hsv_color = np.uint8([[[int(mapped_val), 180 + math.sin(mapped_val) * 180 / (2 * np.pi),
                                        255]]])  # Hue=0 (red), Saturation=255, Value=255
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

                r = int(bgr_color[0])
                b = int(bgr_color[1])
                g = int(bgr_color[2])

                cv2.arrowedLine(img,
                                (x, y),
                                (end_x, end_y),
                                (b, g, r),
                                1,
                                tipLength=0.4)
                                
        average_angle = degree_sum / count / 180
        cv2.putText(img,
                    f"average angle: {average_angle:.4f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Black canvas and center the image
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        x_offset = (screen_w - new_w) // 2
        y_offset = (screen_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        cv2.imshow("Motion Mask", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
