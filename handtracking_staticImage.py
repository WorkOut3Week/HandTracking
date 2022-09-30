import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

Wr = 0
M0 = 9
M3 = 12

# ToDo 1: stack 구현
  # 전처리 1: vector 각도 계산 (방향성도 있어야 됨) -> outliear 1
  # 전처리 2: tracking 실패 시 전처리 -> outlier 2
  # 2.2 
IMAGESTACK = []

def ivector(vec):
  x,y,z = vec
  norm = (x**2+y**2+z**2)**(1/2)
  return [x/norm, y/norm, z/norm]

IMAGE_FILES = [r'C:\Users\37739\Documents\khuthon_2022\HandTracking\static_images\mid.jpg']
              
              # r'C:\Users\37739\Documents\khuthon_2022\HandTracking\static_images\right.jpg',
              # r'C:\Users\37739\Documents\khuthon_2022\HandTracking\static_images\straight.jpg']
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      # print(len(hand_landmarks))
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      # with open(r'C:\Users\37739\Documents\khuthon_2022\HandTracking\static_landmarks\left.txt','a') as f:
      #   for idx, landmark in enumerate(hand_landmarks.landmark):
      #     # print(idx,landmark)
      #     f.write(f"{idx}\n")
      #     f.write(f"{landmark}\n")

      # print(f"0,{hand_landmarks.landmark[0].x}")
      
      Wr2M0 = [hand_landmarks.landmark[M0].x-hand_landmarks.landmark[Wr].x,
              hand_landmarks.landmark[M0].y-hand_landmarks.landmark[Wr].y,
              hand_landmarks.landmark[M0].z-hand_landmarks.landmark[Wr].z]
      Wr2M3 = [hand_landmarks.landmark[M3].x-hand_landmarks.landmark[Wr].x,
              hand_landmarks.landmark[M3].y-hand_landmarks.landmark[Wr].y,
              hand_landmarks.landmark[M3].z-hand_landmarks.landmark[Wr].z]
      M02M3 = [hand_landmarks.landmark[M3].x-hand_landmarks.landmark[M0].x,
              hand_landmarks.landmark[M3].y-hand_landmarks.landmark[M0].y,
              hand_landmarks.landmark[M3].z-hand_landmarks.landmark[M0].z]
      print("---vectors---")
      print(Wr2M0)
      print(Wr2M3)
      print(M02M3)
      print("Ivec")
      print(ivector(Wr2M0))
      print(ivector(Wr2M3))
      print(ivector(M02M3))
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        r'C:\Users\37739\Documents\khuthon_2022\HandTracking\static_images'+"\\" + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
