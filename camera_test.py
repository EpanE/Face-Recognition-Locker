import cv2

def main():
    # Try the first camera device
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera at index 0")
        return

    # Optional: ask for HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera opened")
    print("Press q in the video window to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera test", frame)

        # Wait 1 ms and check for q key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and window closed")

if __name__ == "__main__":
    main()
