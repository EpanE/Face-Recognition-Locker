import cv2
import dlib
import csv
import numpy as np
import time
import threading
import tkinter as tk
import os

from gpiozero import Device, OutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

# Force Pi 5 GPIO backend
Device.pin_factory = LGPIOFactory()

# Relay setup
RELAY_PIN = 17
relay = OutputDevice(RELAY_PIN, active_high=False, initial_value=True)

# Recognition parameters
MATCH_THRESHOLD = 0.55
REQUIRED_VISIBLE_SECONDS = 2.0       # how long the same person must be seen
RELAY_ACTIVE_TIME = 3.0              # seconds relay stays ON
COOLDOWN_TIME = 3.0                  # seconds after relay OFF before next unlock

# State variables
last_name = "Unknown"
same_name_start_time = None
relay_locked = False
relay_off_time = 0.0
cooldown_until = 0.0

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = BASE + "/data/data_dlib/shape_predictor_68_face_landmarks.dat"
FACE_MODEL_PATH = BASE + "/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
CSV_PATH = BASE + "/data/features_all.csv"

# Load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_MODEL_PATH)

# Load CSV features
labels = []
descriptors = []

with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 129:
            continue
        labels.append(row[0])
        descriptors.append(np.array([float(x) for x in row[1:129]], dtype=np.float32))

descriptors = np.array(descriptors)
print(f"Loaded {len(descriptors)} face descriptors")


def match_face(face_descriptor):
    distances = np.linalg.norm(descriptors - face_descriptor, axis=1)
    idx = np.argmin(distances)
    dist = distances[idx]
    if dist < MATCH_THRESHOLD:
        return labels[idx], dist
    return "Unknown", dist


def set_relay(on):
    global relay_locked, relay_off_time, cooldown_until
    now = time.time()

    if on:
        print("Relay ON")
        relay.on()
        relay_locked = True
        relay_off_time = now + RELAY_ACTIVE_TIME
        cooldown_until = relay_off_time + COOLDOWN_TIME
    else:
        print("Relay OFF")
        relay.off()
        relay_locked = False


def video_loop():
    global last_name, same_name_start_time
    global relay_locked, relay_off_time, cooldown_until

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("System running. Press q in video window to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        current_name = "Unknown"
        now = time.time()

        # Auto switch off relay after RELAY_ACTIVE_TIME
        if relay_locked and now > relay_off_time:
            set_relay(False)

        if faces:
            # use largest face
            face = max(
                faces,
                key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top())
            )

            shape = sp(frame, face)
            aligned = dlib.get_face_chip(frame, shape)
            descriptor = np.array(facerec.compute_face_descriptor(aligned))

            name, dist = match_face(descriptor)
            current_name = name

            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({dist:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # time based same name logic
            if name != "Unknown":
                if name == last_name:
                    # still same person
                    if same_name_start_time is None:
                        same_name_start_time = now
                    visible_duration = now - same_name_start_time
                else:
                    # new person
                    last_name = name
                    same_name_start_time = now
                    visible_duration = 0.0

                # condition to unlock
                if (
                    visible_duration >= REQUIRED_VISIBLE_SECONDS
                    and not relay_locked
                    and now > cooldown_until
                ):
                    print(f"Access granted to {name}"
                          f" after {visible_duration:.1f} seconds")
                    set_relay(True)
                    # reset timer so next unlock needs new continuous view
                    same_name_start_time = None

            else:
                # name is Unknown
                last_name = "Unknown"
                same_name_start_time = None

        else:
            current_name = "No face"
            last_name = "Unknown"
            same_name_start_time = None

        # overlay info
        cv2.putText(
            frame,
            f"Current: {current_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Face Access Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    set_relay(False)
    cap.release()
    cv2.destroyAllWindows()


def ui_loop():
    def manual_unlock():
        now = time.time()
        if now > cooldown_until:
            print("Manual unlock triggered")
            set_relay(True)
        else:
            print("Cooldown active, manual unlock blocked")

    win = tk.Tk()
    win.title("Door Control")

    tk.Label(
        win,
        text="Face Recognition Door System",
        font=("Arial", 16),
    ).pack(pady=10)

    tk.Button(
        win,
        text="Unlock Door",
        font=("Arial", 16),
        command=manual_unlock,
    ).pack(pady=20)

    tk.Label(win, text="Close window to exit").pack()

    win.mainloop()


if __name__ == "__main__":
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()
    ui_loop()
    # when UI closes, program ends, video thread stops on next loop
