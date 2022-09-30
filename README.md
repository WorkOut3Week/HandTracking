# HandTracking

## HandTracking_staticImage.py

* line 22: `print('Handedness:', results.multi_handedness)`

* output: 
```
Handedness: [classification {
  index: 0 # 손 인덱스
  score: 0.9864261746406555 # 손인지에 대한 confidence score
  label: "Left" # 왼손 vs 오른손
}
```
* line 27: `print('hand_landmarks:', hand_landmarks)`
```
hand_landmarks: landmark {
  x: 0.33169272541999817
  y: 0.5325332283973694
  z: 2.984734237543307e-07
}

.
.
.

landmark {
  x: 0.37709441781044006
  y: 0.1419113129377365
  z: -0.03787199780344963
}
```
손 1개에 21개
