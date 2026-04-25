# debug_coords.py - run this on each node to sanity check calibration
import json, numpy as np, pyrealsense2 as rs, cv2, os

CALIB_FILE = os.path.expanduser("~/supermesh_calib.json")

with open(CALIB_FILE) as f:
    data = json.load(f)

R_inv = np.array(data["R_inv"])
tvec  = np.array(data["tvec"])

print("=== CALIBRATION SANITY CHECK ===")
print(f"tvec magnitude: {np.linalg.norm(tvec):.2f}m  (should ≈ camera distance to board)")
print(f"tvec raw: {tvec.flatten()}")
print(f"R_inv determinant: {np.linalg.det(R_inv):.4f}  (should be exactly 1.0)")
print()

# Stream a few frames and print raw world coords
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

print("Standing at board center - world coords should be near 0,0,0")
print("Walk to a corner - X and Z should change, Y should stay near 0")
print("Ctrl+C to stop\n")

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        cx, cy = 320, 400  # sample center-lower of frame

        patch = depth_image[cy-10:cy+10, cx-15:cx+15]
        valid = patch[patch > 0]
        if not len(valid):
            continue

        depth_m = float(np.median(valid)) * depth_scale
        if not (0.3 < depth_m < 8.0):
            continue

        cam_pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
        cam_mat = np.array([[cam_pt[0]], [cam_pt[1]], [cam_pt[2]]])
        world = R_inv @ (cam_mat - tvec)

        print(f"X:{world[0][0]:+.2f}  Y:{world[1][0]:+.2f}  Z:{world[2][0]:+.2f}  depth:{depth_m:.2f}m", end='\r')

except KeyboardInterrupt:
    pipeline.stop()
    print("\nDone.")