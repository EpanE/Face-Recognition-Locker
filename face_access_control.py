import cv2
import dlib
import numpy as np
import csv
import time
import threading
from gpiozero import OutputDevice
import tkinter as tk

# GPIO setup using gpiozero
RELAY_PIN = 17  # BCM numbering

# Many 5V relay modules are active LOW: input LOW = relay ON
# active_high=False means "on()" will drive the pin LOW.
relay = OutputDevice(RELAY_PIN, active_high=False, initial_value=True)

# Face recognition parameters
CONSECUTIVE_FRAMES_REQUIRED = 5
RELAY_ON_DURATION = 5.0       # seconds
COOLDOWN_DURATION = 3.0       # seconds
MATCH_THRESHOLD = 0.5         # adjust based on your data

# Paths to models and CSV
PREDICTOR_PATH = "data/data_dlib/shape_predictor_5_face_landmarks.dat"
FACE_RECOG_MODEL_PATH = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
CSV_FEATURES_PATH = "data/features_all.csv"

# Load dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

# Load known face descriptors from CSV
known_descriptors = []
known_labels = []

with open(CSV_FEATURES_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 129:
            continue
        label = row[0]
        descriptor = np.array([float(x) for x in row[1:129]], dtype=np.float32)
        known_labels.append(label)
        known_descriptors.append(descriptor)

known_descriptors = np.array(known_descriptors)

# State variables for recognition and relay
current_name = "Unknown"
last_name = "Unknown"
same_name_count = 0

relay_locked = False
relay_on_until = 0.0
cooldown_until = 0.0

# Thread control
stop_flag = False


def set_relay(on):
    """Turn relay on/off and manage timers."""
    global relay_locked, relay_on_until, cooldown_until
    now = time.time()
    if on:
        relay.on()   # active_low module: on() = LOW = relay energised
        relay_locked = True
        relay_on_until = now + RELAY_ON_DURATION
        cooldown_until = relay_on_until + COOLDOWN_DURATION
        print("Relay ON until", relay_on_until)
    else:
        relay.off()  # off() = HIGH = relay released
        relay_locked = False
        print("Relay OFF")


def match_face(face_descriptor):
    if len(known_descriptors) == 0:
        return "Unknown", 1e9

    distances = np.linalg.norm(known_descriptors - face_descriptor, axis=1)
    idx = np.argmin(distances)
    min_dist = distances[idx]
    if min_dist < MATCH_THRESHOLD:
        return known_labels[idx], min_dist
    else:
        return "Unknown", min_dist


def video_loop():
    global current_name, last_name, same_name_count
    global relay_locked, relay_on_until, cooldown_until
    global stop_flag

    cap = cv2.VideoCapture(0)  # change index if needed

    if not cap.isOpened():
        print("Cannot open camera")
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        current_name = "Unknown"

        if len(faces) > 0:
            # Use the largest face
            areas = []
            for f in faces:
                areas.append((f.right() - f.left()) * (f.bottom() - f.top()))
            largest_idx = int(np.argmax(areas))
            face_rect = faces[largest_idx]

            shape = shape_predictor(frame, face_rect)
            face_chip = dlib.get_face_chip(frame, shape)
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(face_chip))

            name, dist = match_face(face_descriptor)
            current_name = name

            # Draw box and name
            x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{name} {dist:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Consecutive frame logic
            if name == last_name and name != "Unknown":
                same_name_count += 1
            else:
                same_name_count = 1
                last_name = name

            now = time.time()

            # Handle relay timers
            if relay_locked and now > relay_on_until:
                set_relay(False)

            # Trigger relay if conditions met
            if (name != "Unknown"
                and same_name_count >= CONSECUTIVE_FRAMES_REQUIRED
                and not relay_locked
                and now > cooldown_until):
                set_relay(True)

        else:
            current_name = "Unknown"
            same_name_count = 0
            last_name = "Unknown"
            now = time.time()
            if relay_locked and now > relay_on_until:
                set_relay(False)

        # Show current name on screen even if no face box
        cv2.putText(frame, f"Current: {current_name}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Face Access Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    set_relay(False)


def ui_loop():
    def on_unlock_click():
        now = time.time()
        if now > cooldown_until:
            print("Manual unlock button pressed")
            set_relay(True)

    window = tk.Tk()
    window.title("Door Control")

    label = tk.Label(window, text="Face Access System", font=("Arial", 16))
    label.pack(pady=10)

    button = tk.Button(window, text="Unlock door", font=("Arial", 14),
                       command=on_unlock_click)
    button.pack(pady=20)

    info_label = tk.Label(window, text="Close window or press Q in video to exit")
    info_label.pack(pady=10)

    def on_close():
        global stop_flag
        stop_flag = True
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)
    window.mainloop()


if __name__ == "__main__":
    try:
        t_video = threading.Thread(target=video_loop, daemon=True)
        t_video.start()
        ui_loop()
    finally:
        stop_flag = True
        set_relay(False)
