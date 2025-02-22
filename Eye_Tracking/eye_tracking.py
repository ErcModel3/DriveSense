import cv2
import sys
from matplotlib import pyplot as plt

def detect(img, model):
    # OpenCV opens images as BRG 
    # but we want it as RGB We'll 
    # also need a grayscale version
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use minSize because for not 
    # bothering with extra-small
    # dots that would look like STOP signs
    data = cv2.CascadeClassifier(model)
    boxes = data.detectMultiScale(img_gray, minSize =(20, 20))
    if boxes is None:
        return []
    return boxes
          
def tracking():
    # Can change the parameter (0) to whatever camera the device needs to pull from
    cap = cv2.VideoCapture(0)

    # Initialize tracker
    NUM_TRACKERS = 2
    trackers = []
    for i in range(NUM_TRACKERS):
        trackers.append(cv2.TrackerKCF_create())
    init_box0 = []  # Assuming init_box is the bounding box of detected object
    init_box1 = []

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit()

    print("Camera opened successfully! Press 'q' to quit")

    while len(init_box0) == 0 or len(init_box1) == 0:
        ret, frame = cap.read()
        if ret:
            print("balls")

            init_box0 = detect(frame, 'Eye_Tracking/haarcascade_lefteye_2splits.xml')
            init_box1 = detect(frame, 'Eye_Tracking/haarcascade_righteye_2splits.xml')
            
            if len(init_box0) > 0 and len(init_box1) > 0:
                trackers[0].init(frame, init_box0[0])
                trackers[1].init(frame, init_box1[0])

    # ret, frame = cap.read()
    # if ret:
    #     init_boxes = detect(frame, 'haarcascade_eye.xml')
    #     for i in range(NUM_TRACKERS):
    #         trackers[i].init(frame, init_boxes[i])

    consec_failures = 0

    while True:
        ret, frame = cap.read()

        if consec_failures > 50:
            init_box0 = detect(frame, 'Eye_Tracking/haarcascade_lefteye_2splits.xml')
            init_box1 = detect(frame, 'Eye_Tracking/haarcascade_righteye_2splits.xml')
            if len(init_box0) > 0 and len(init_box1) > 0:
                trackers[0] = cv2.TrackerKCF_create()
                trackers[0].init(frame, init_box0[0])
                trackers[1] = cv2.TrackerKCF_create()
                trackers[1].init(frame, init_box1[0])
                consec_failures = 0


        # Checks to see if the frame is capturing correctly
        if not ret:
            print("Error: Can't receive frame from camera")
            break
        
            
        for tracker in trackers:
            ret, bbox = tracker.update(frame)
            if ret:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            else:
                consec_failures += 1
                cv2.putText(frame, "Tracking failure: " + str(consec_failures), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tracking()

