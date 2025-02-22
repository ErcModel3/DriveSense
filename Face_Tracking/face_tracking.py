import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def pt_from_lm(lm):
  return np.array([lm.x, lm.y, lm.z])


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
with mp_face_mesh.FaceMesh(
  max_num_faces=3,
  refine_landmarks=True,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as face_mesh:
  index = 0
  pool = np.zeros(int(fps * 3))
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        shape = image.shape
        pt1 = pt_from_lm(face_landmarks.landmark[1])
        pt2 = pt_from_lm(face_landmarks.landmark[2])
        pt3 = pt_from_lm(face_landmarks.landmark[168])
        triangle_cnt = np.array([(pt1[0] * shape[1], pt1[1] * shape[0]),
                                 (pt2[0] * shape[1], pt2[1] * shape[0]),
                                 (pt3[0] * shape[1], pt3[1] * shape[0])]).astype(np.int32)
        cv2.drawContours(image, [triangle_cnt], 0, (0,255,0), -1)
        forward_vector = pt1 - (pt2 + (pt3 - pt2) / 4)
        forward_vector /= np.linalg.norm(forward_vector)
        
        # angle of face from xz plane (0 is looking straight ahead, negative is looking down)
        angle_from_xz = np.arctan(forward_vector[1] / forward_vector[2])
        # angle of face from yz plane
        angle_from_yz = np.arctan(forward_vector[0] / forward_vector[2])

        # calculate running avg of the angle from the xz plane
        pool[index] = angle_from_xz
        index += 1
        index %= len(pool)
        if np.all(pool != 0):
          angle = np.mean(pool)
          if angle < -0.55:
            cv2.putText(image, "LOOK UP! {0:.3f}".format(angle), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()