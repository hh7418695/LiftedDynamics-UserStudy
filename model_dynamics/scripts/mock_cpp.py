"""
mock_cpp.py - Simulates the C++ UDP client (TouchX haptic device).

Usage:
    python mock_cpp.py

Controls:
    b - simulate stylus button press (toggle start/stop rendering)
    q - quit

Sends JSON: {"position": [x, y, z], "timestamp": <ms>}
Stop signal: {"position": [0, 0, 0], "timestamp": -1}
"""

import json
import socket
import threading
import time
import numpy as np

SERVER_IP = "127.0.0.1"
SERVER_PORT = 12312
SEND_RATE_HZ = 100  # mock sends at 100 Hz (real device is 1kHz)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

rendering = False       # True while stylus "button" is held active
running = True          # False to quit the mock
start_time = time.time()


def _random_position() -> list:
    """Generate slowly drifting random position in mm (TouchX coordinate space)."""
    t = time.time() - start_time
    x = 20.0 * np.sin(0.3 * t) + np.random.normal(0, 0.5)
    y = 15.0 * np.cos(0.2 * t) + np.random.normal(0, 0.5)
    z = 10.0 * np.sin(0.5 * t) + np.random.normal(0, 0.5)
    return [round(x, 3), round(y, 3), round(z, 3)]


def _send_loop():
    """Continuously send position packets while rendering is active."""
    global rendering, running
    warmup_sent = False

    while running:
        if rendering:
            # Send 5 warm-up packets first (zero position, positive timestamp)
            if not warmup_sent:
                for _ in range(5):
                    pkt = json.dumps({"position": [0.0, 0.0, 0.0],
                                      "timestamp": int((time.time() - start_time) * 1000)})
                    sock.sendto(pkt.encode(), (SERVER_IP, SERVER_PORT))
                    time.sleep(1.0 / SEND_RATE_HZ)
                warmup_sent = True

            pos = _random_position()
            ts = int((time.time() - start_time) * 1000)
            pkt = json.dumps({"position": pos, "timestamp": ts})
            sock.sendto(pkt.encode(), (SERVER_IP, SERVER_PORT))
            time.sleep(1.0 / SEND_RATE_HZ)
        else:
            warmup_sent = False
            time.sleep(0.01)


def _keyboard_loop():
    """Read keyboard input to toggle rendering or quit."""
    global rendering, running
    print("[mock_cpp] Controls: b = toggle button | q = quit")
    while running:
        try:
            key = input().strip().lower()
        except EOFError:
            break

        if key == "b":
            if not rendering:
                rendering = True
                print("[mock_cpp] Button pressed → START rendering")
            else:
                rendering = False
                # Send stop signal
                stop_pkt = json.dumps({"position": [0.0, 0.0, 0.0], "timestamp": -1})
                sock.sendto(stop_pkt.encode(), (SERVER_IP, SERVER_PORT))
                print("[mock_cpp] Button pressed → STOP rendering (stop signal sent)")
        elif key == "q":
            if rendering:
                stop_pkt = json.dumps({"position": [0.0, 0.0, 0.0], "timestamp": -1})
                sock.sendto(stop_pkt.encode(), (SERVER_IP, SERVER_PORT))
            running = False
            print("[mock_cpp] Quitting.")


if __name__ == "__main__":
    send_thread = threading.Thread(target=_send_loop, daemon=True)
    send_thread.start()
    _keyboard_loop()
