import cv2
import os
import time

# Configuration
PERSON_NAME = "Dinie"  # Change this to the person's name
DATASET_PATH = "dataset"  # Base directory for datasets
MAX_IMAGES = 200  # Maximum number of images to capture
CAPTURE_DELAY = 0.05  # Delay between captures in seconds

# Create dataset directory structure
person_folder = os.path.join(DATASET_PATH, PERSON_NAME)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)
    print(f"Created directory: {person_folder}")

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(r"VID20251212172052.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Variables for tracking
image_count = 0
last_capture_time = 0
collection_complete = False

print("Starting face dataset collection...")
print(f"Press 'q' to quit early")
print(f"Target: {MAX_IMAGES} images for {PERSON_NAME}")

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
            
            print(f"Captured: {image_count}/{MAX_IMAGES}")
            
            # Check if collection is complete
            if image_count >= MAX_IMAGES:
                collection_complete = True
                print(f"\n✓ Dataset collection completed!")
                print(f"Saved {image_count} images to: {person_folder}")
    
    # Add text overlay with information
    # Person name
    cv2.putText(display_frame, f"Person: {PERSON_NAME}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Capture progress
    progress_text = f"Captured: {image_count} / {MAX_IMAGES}"
    color = (0, 255, 0) if not collection_complete else (0, 255, 255)
    cv2.putText(display_frame, progress_text, 
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Completion message
    if collection_complete:
        cv2.putText(display_frame, "Dataset Collection Completed!", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to exit", 
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show instructions
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

print(f"\nFinal Summary:")
print(f"Total images captured: {image_count}")
print(f"Saved in: {person_folder}")
print(f"Ready for training!")