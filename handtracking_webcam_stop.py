import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

Wr = 0
M0 = 9
M3 = 12

def ivector(vec):
  x,y,z = vec
  norm = (x**2+y**2+z**2)**(1/2)
  return [x/norm, y/norm, z/norm]

image_num = 0
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    stime = time.time()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # if results.multi_handedness is not None:
    #   for mlh in results.multi_handedness:
    #     print(mlh.classification[0].score)
        # if mlh is not None and mlh.classification.label == "Right":
          # print("yes")
      # print(results.multi_handedness[0].classification) 
      # if results.multi_handedness.
      # cv2.imwrite(r"C:\Users\37739\Documents\khuthon_2022\HandTracking\conf_images"+f"\{}"+)
    # print(results.multi_handedness)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        Wr2M3 = [hand_landmarks.landmark[M3].x-hand_landmarks.landmark[Wr].x,
        hand_landmarks.landmark[M3].y-hand_landmarks.landmark[Wr].y,
        hand_landmarks.landmark[M3].z-hand_landmarks.landmark[Wr].z]
        i_Wr2M3 = ivector(Wr2M3)
        if abs(i_Wr2M3[2]/i_Wr2M3[0]) >= 2.14:
          print("Go Straight")
        print(i_Wr2M3)

        # print(hand_landmarks)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if results.multi_hand_landmarks:
      for k, (mlh, hand_landmarks) in enumerate(zip(results.multi_handedness, results.multi_hand_landmarks)):
        # print(k)
        if  mlh.classification[0].label == "Right":
          pass
          # print(hand_landmarks)
          # print(mlh.classification[0])
          # if mlh.classification[0].score < 0.6:
          #   assert image_num <10
          #   image_num+=1
          #   cv2.imwrite(r"C:\Users\37739\Documents\khuthon_2022\HandTracking\conf_images"+f"\{mlh.classification[0].score}"+'.jpg',image)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    etime =time.time()
    # print(etime-stime)
cap.release()