"""
SuperMesh Movement Simulator
Sends fake fused person OSC messages to PureData on port 9005,
mimicking exactly what master_enhanced.py would send.

Usage:
    python supermesh_sim.py [mode]

Modes:
    circle   - person walks in a slow circle (default)
    figure8  - figure-of-eight path
    random   - random wandering
    manual   - control position with keyboard (WASD, Q to quit)

Dependencies:
    pip install python-osc
"""

import time
import math
import sys
import random
from pythonosc.udp_client import SimpleUDPClient

# --- CONFIG --- match these to your master_enhanced.py settings
PD_IP   = "127.0.0.1"
PD_PORT = 9005
SEND_RATE_HZ = 20       # how many times per second to send
ROOM_RADIUS  = 3.0      # meters — keeps movement within a realistic room size

client = SimpleUDPClient(PD_IP, PD_PORT)
interval = 1.0 / SEND_RATE_HZ

def send(x, z, person_id=1):
    """Send a fused position exactly as master_enhanced.py does."""
    client.send_message(f"/person/{person_id}", [float(x), float(z)])
    print(f"  /person/{person_id}  x={x:.3f}  z={z:.3f}")

# --- MODES ---

def mode_circle():
    print("Mode: CIRCLE  |  Ctrl+C to stop")
    t = 0.0
    while True:
        x = math.cos(t) * ROOM_RADIUS
        z = math.sin(t) * ROOM_RADIUS
        send(x, z)
        t += 0.02   # speed of orbit — increase to walk faster
        time.sleep(interval)

def mode_figure8():
    print("Mode: FIGURE-8  |  Ctrl+C to stop")
    t = 0.0
    while True:
        x = math.sin(t) * ROOM_RADIUS
        z = math.sin(t * 2) * (ROOM_RADIUS / 2)
        send(x, z)
        t += 0.02
        time.sleep(interval)

def mode_random():
    print("Mode: RANDOM WANDER  |  Ctrl+C to stop")
    x, z = 0.0, 0.0
    vx, vz = 0.05, 0.05
    while True:
        # Random nudge to velocity
        vx += random.uniform(-0.02, 0.02)
        vz += random.uniform(-0.02, 0.02)
        # Clamp velocity
        vx = max(-0.15, min(0.15, vx))
        vz = max(-0.15, min(0.15, vz))
        # Bounce off room boundaries
        x += vx
        z += vz
        if abs(x) > ROOM_RADIUS: vx *= -1
        if abs(z) > ROOM_RADIUS: vz *= -1
        x = max(-ROOM_RADIUS, min(ROOM_RADIUS, x))
        z = max(-ROOM_RADIUS, min(ROOM_RADIUS, z))
        send(x, z)
        time.sleep(interval)

def mode_manual():
    """WASD keyboard control. Requires 'keyboard' package: pip install keyboard"""
    try:
        import keyboard
    except ImportError:
        print("Manual mode requires the 'keyboard' package: pip install keyboard")
        sys.exit(1)

    print("Mode: MANUAL  |  WASD to move, Q to quit")
    x, z = 0.0, 0.0
    step = 0.1

    while True:
        if keyboard.is_pressed('q'):
            print("Quit.")
            break
        if keyboard.is_pressed('w'): z -= step
        if keyboard.is_pressed('s'): z += step
        if keyboard.is_pressed('a'): x -= step
        if keyboard.is_pressed('d'): x += step
        x = max(-ROOM_RADIUS, min(ROOM_RADIUS, x))
        z = max(-ROOM_RADIUS, min(ROOM_RADIUS, z))
        send(x, z)
        time.sleep(interval)

# --- MAIN ---
modes = {
    "circle":  mode_circle,
    "figure8": mode_figure8,
    "random":  mode_random,
    "manual":  mode_manual,
}

mode = sys.argv[1] if len(sys.argv) > 1 else "circle"

if mode not in modes:
    print(f"Unknown mode '{mode}'. Choose from: {', '.join(modes.keys())}")
    sys.exit(1)

print(f"SuperMesh Simulator → {PD_IP}:{PD_PORT}")
print(f"Range: x and z both within -{ROOM_RADIUS} to +{ROOM_RADIUS} meters")
print()

try:
    modes[mode]()
except KeyboardInterrupt:
    print("\nStopped.")
