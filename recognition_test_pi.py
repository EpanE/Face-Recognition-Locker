import cv2
import dlib
import csv
import numpy as np
import os

# Paths for models and features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(
    BASE_DIR,
    "data",
    "data_dlib",
    "shape_predictor_68_face_landmarks.dat",
)
FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "data_dlib",
    "dlib_face_recognition_resnet_model_v1.dat",
)
CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features_all.csv",
)

# Load dlib models
print("Loading dlib models...")
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_MODEL_PATH)
print("Models loaded")

# Load known features from CSV
print("Loading known faces from CSV:", CSV_PATH)
labels = []
descriptors = []

if not os.path.exists(CSV_PATH):
    print("ERROR: CSV file not found at", CSV_PATH)
    exit(1)

with open(CSV_PATH, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 129:
            continue
        label = row[0]
        vector = np.array([float(x) for x in row[1:129]], dtype=np.float32)
        labels.append(label)
        descriptors.append(vector)

if len(descriptors) == 0:
    print("No descriptors loaded from CSV. Check your features file.")
    exit(1)

descriptors = np.array(descriptors)
print(f"Loaded {len(descriptors)} known face descriptors")

# Matching parameters
MATCH_THRESHOLD = 0.55   # you can tune this later

def match_face(descriptor):
    """Return (name, distance) for closest match."""
    distances = np.linalg.norm(descriptors - descriptor, axis=1)
    idx = np.argmin(distances)
    best_distance = distances[idx]
    if best_distance < MATCH_THRESHOLD:
        return labels[idx], best_distance
    return "Unknown", best_distance

def main():
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("ERROR: Cannot open camera at index 0")
        return

    # Optional resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Recognition test running")
    print("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            shape = shape_predictor(frame, face)
            face_chip = dlib.get_face_chip(frame, shape)

            descriptor = np.array(
                face_rec_model.compute_face_descriptor(face_chip),
                dtype=np.float32,
            )

            name, dist = match_face(descriptor)

            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{name} ({dist:.2f})"
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            "Press q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Recognition test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released, window closed")

if __name__ == "__main__":
    main()
