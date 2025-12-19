import cv2
import os
import time
import dlib
import csv
import numpy as np

# ======================== Configuration ========================
PERSON_NAME = "Nik"  # Change this to the person's name
DATASET_PATH = "data/data_faces_from_camera"  # Base directory for datasets
MAX_IMAGES = 200  # Maximum number of images to capture
CAPTURE_DELAY = 0.4  # Delay between captures in seconds

# Paths for Dlib models (required for feature extraction)
SHAPE_PREDICTOR_PATH = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "data/data_dlib/dlib_face_recognition_resnet_model_v1.dat"
FEATURES_CSV_PATH = "data/features_all.csv"

# ======================== Setup ========================
# Create dataset directory structure
person_folder = os.path.join(DATASET_PATH, f"person_{PERSON_NAME}")
if not os.path.exists(person_folder):
    os.makedirs(person_folder)
    print(f"Created directory: {person_folder}")

# Create data_dlib directory if it doesn't exist
os.makedirs("data/data_dlib", exist_ok=True)

# Load Haar Cascade face detector for capture
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ======================== Feature Extraction Functions ========================

def check_dlib_models():
    """Check if required Dlib models exist"""
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"\n  Warning: Shape predictor not found at {SHAPE_PREDICTOR_PATH}")
        print("Feature extraction will be skipped.")
        return False
    if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
        print(f"\n  Warning: Face recognition model not found at {FACE_RECOGNITION_MODEL_PATH}")
        print("Feature extraction will be skipped.")
        return False
    return True

def return_128d_features(path_img, detector, predictor, face_reco_model):
    """Return 128D features for single image"""
    img_rd = cv2.imread(path_img)
    if img_rd is None:
        print(f"Warning: Could not read image {path_img}")
        return 0
    
    faces = detector(img_rd, 1)
    
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        return face_descriptor
    else:
        print(f"Warning: No face detected in {path_img}")
        return 0

def return_features_mean_personX(path_face_personX, detector, predictor, face_reco_model):
    """Return the mean value of 128D face descriptor for person X"""
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    
    if photos_list:
        print(f"\nProcessing {len(photos_list)} images for feature extraction...")
        for i, photo in enumerate(photos_list):
            if i % 20 == 0:  # Progress update every 20 images
                print(f"Processing image {i+1}/{len(photos_list)}...")
            
            photo_path = os.path.join(path_face_personX, photo)
            features_128d = return_128d_features(photo_path, detector, predictor, face_reco_model)
            
            if features_128d != 0:
                features_list_personX.append(features_128d)
    else:
        print(f"Warning: No images in {path_face_personX}/")
    
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    
    return features_mean_personX

def extract_features_to_csv():
    """Extract features from all collected face images and save to CSV"""
    print("\n" + "="*60)
    print("Starting Feature Extraction Process...")
    print("="*60)
    
    # Check if Dlib models exist
    if not check_dlib_models():
        print("\n Feature extraction skipped due to missing Dlib models.")
        print("\nTo enable feature extraction, download the following files:")
        print("1. shape_predictor_68_face_landmarks.dat")
        print("2. dlib_face_recognition_resnet_model_v1.dat")
        print(f"Place them in: {os.path.abspath('data/data_dlib/')}")
        return False
    
    try:
        # Initialize Dlib models
        print("\nLoading Dlib models...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        face_reco_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
        print("✓ Dlib models loaded successfully")
        
        # Get list of all persons in dataset
        person_list = os.listdir(DATASET_PATH)
        person_list.sort()
        
        if not person_list:
            print("Warning: No person folders found in dataset")
            return False
        
        print(f"\nFound {len(person_list)} person(s) in dataset: {', '.join(person_list)}")
        
        # Extract features and save to CSV
        with open(FEATURES_CSV_PATH, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            for person in person_list:
                person_path = os.path.join(DATASET_PATH, person)
                if not os.path.isdir(person_path):
                    continue
                
                print(f"\n--- Processing: {person} ---")
                features_mean_personX = return_features_mean_personX(
                    person_path, detector, predictor, face_reco_model
                )
                
                # Extract person name from folder name
                if len(person.split('_', 1)) == 2:
                    person_name = person.split('_', 1)[1]  # "person_Nik" -> "Nik"
                else:
                    person_name = person
                
                # Insert person name at the beginning (129D total)
                features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
                writer.writerow(features_mean_personX)
                print(f"✓ Features extracted for {person_name}")
        
        print(f"\n{'='*60}")
        print(f"✓ Feature extraction completed!")
        print(f"✓ Features saved to: {os.path.abspath(FEATURES_CSV_PATH)}")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\n Error during feature extraction: {str(e)}")
        return False

# ======================== Face Collection Process ========================

def collect_face_images():
    """Main function to collect face images from webcam"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Variables for tracking
    image_count = 0
    last_capture_time = 0
    collection_complete = False
    
    print("\n" + "="*60)
    print("Starting Face Dataset Collection...")
    print("="*60)
    print(f"Person: {PERSON_NAME}")
    print(f"Target: {MAX_IMAGES} images")
    print(f"Save location: {person_folder}")
    print(f"Press 'q' to quit early")
    print("="*60 + "\n")
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Process detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check if we should capture an image
            current_time = time.time()
            if (not collection_complete and 
                image_count < MAX_IMAGES and 
                (current_time - last_capture_time) >= CAPTURE_DELAY):
                
                # Crop the face region
                face_crop = frame[y:y+h, x:x+w]
                
                # Save the cropped face image
                image_filename = os.path.join(person_folder, f"{PERSON_NAME}_{image_count+1}.jpg")
                cv2.imwrite(image_filename, face_crop)
                
                # Update counters
                image_count += 1
                last_capture_time = current_time
                
                if image_count % 10 == 0:  # Print every 10 images
                    print(f"Progress: {image_count}/{MAX_IMAGES} images captured")
                
                # Check if collection is complete
                if image_count >= MAX_IMAGES:
                    collection_complete = True
                    print(f"\n{'='*60}")
                    print(f"✓ Dataset collection completed!")
                    print(f"✓ Saved {image_count} images to: {person_folder}")
                    print(f"{'='*60}")
        
        # Add text overlay with information
        cv2.putText(display_frame, f"Person: {PERSON_NAME}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        progress_text = f"Captured: {image_count} / {MAX_IMAGES}"
        color = (0, 255, 0) if not collection_complete else (0, 255, 255)
        cv2.putText(display_frame, progress_text, 
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        if collection_complete:
            cv2.putText(display_frame, "Dataset Collection Completed!", 
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to continue to feature extraction", 
                        (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(display_frame, "Press 'q' to quit", 
                    (10, display_frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Display the frame
        cv2.imshow('Face Dataset Collection', display_frame)
        
        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCollection stopped by user")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    return image_count > 0

# ======================== Main Program ========================

def main():
    """Main program flow"""
    print("\n" + "="*60)
    print("FACE RECOGNITION DATASET COLLECTION & FEATURE EXTRACTION")
    print("="*60)
    
    # Step 1: Collect face images
    success = collect_face_images()
    
    if not success:
        print("\n No images were collected. Exiting...")
        return
    
    # Step 2: Extract features to CSV
    print("\nProceeding to feature extraction...")
    time.sleep(1)  # Brief pause before starting feature extraction
    
    extract_features_to_csv()
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED!")
    print("="*60)
    print(f"\n✓ Face images saved in: {os.path.abspath(person_folder)}")
    print(f"✓ Features CSV saved in: {os.path.abspath(FEATURES_CSV_PATH)}")
    print("\nYour dataset is ready for training a face recognition model!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()