import cv2
import sys


def start_camera():
    # Can change the parameter (0) to whatever camera the device needs to pull from
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit()

    print("Camera opened successfully! Press 'q' to quit")

    while True:
        ret, frame = cap.read()

        # Checks to see if the frame is capturing correctly
        if not ret:
            print("Error: Can't receive frame from camera")
            break

        cv2.imshow('Live Camera Feed', frame)

        # Wait for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
