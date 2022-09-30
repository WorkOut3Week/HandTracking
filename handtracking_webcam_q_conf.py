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


Queue = []
CONF_THRESHOLD = 0.6
Q_NUM =5

# image_num = 0
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  pre_q = None
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

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 손이 하나 이상 발견하면 첫줄 어팬드, 그렇지 않으면 None 어팬드
    if results.multi_hand_landmarks:
      # print(len(results.multi_handedness))
      Queue.append([results.multi_handedness, results.multi_hand_landmarks])
    else:
      Queue.append(None)

    if len(Queue) > Q_NUM:
      q = Queue.pop(0)
      # print(q)
      pre_q = q
      if q is not None and len(q)!=0:
          for k, (mlh, hand_landmarks) in enumerate(zip(q[0],q[1])):
            # print(list(q))
            if len(q[0])==1 and mlh.classification[0].score < CONF_THRESHOLD:
              if pre_q is not None and len(list(pre_q))==1:
                pre_mlh, pre_hand_landmarks = pre_q
                if pre_mlh.classification[0].score >= CONF_THRESHOLD:
                  hand_landmarks = pre_hand_landmarks 
            Wr2M3 = [hand_landmarks.landmark[M3].x-hand_landmarks.landmark[Wr].x,
            hand_landmarks.landmark[M3].y-hand_landmarks.landmark[Wr].y,
            hand_landmarks.landmark[M3].z-hand_landmarks.landmark[Wr].z]
            i_Wr2M3 = ivector(Wr2M3)
            if abs(i_Wr2M3[2]/i_Wr2M3[0]) >= 2.14:
              print("Go Straight") 
            mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    etime =time.time()
    # print(etime-stime)
cap.release()