import dlib
import os

PREDICTOR_PATH = "data/data_dlib/shape_predictor_5_face_landmarks.dat"
FACE_RECOG_MODEL_PATH = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"

print("Checking model files exist:")
print("  Predictor file:", PREDICTOR_PATH, "exists:", os.path.exists(PREDICTOR_PATH))
print("  Face recog file:", FACE_RECOG_MODEL_PATH, "exists:", os.path.exists(FACE_RECOG_MODEL_PATH))

# Try to load them
try:
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_rec_model = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)
    print("Models loaded successfully")
except Exception as e:
    print("Error while loading models:")
    print(e)
