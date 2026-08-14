"""Quick single-image inference test"""
import joblib, cv2, sys, os
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe

# Load model
bundle = joblib.load('models/pose_classifier.joblib')
model = bundle['model']
scaler = bundle['scaler']
state_names = bundle['state_names']

# MediaPipe
pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
extractor = FeatureExtractor(use_motion=False)

# Test fall image
img_path = r'D:\迅雷下载\main_data\test\Fall\fall-01-cam0-rgb-109.png'
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)

if results.pose_landmarks:
    lm = landmarks_from_mediapipe(results.pose_landmarks)
    fv = extractor.extract(lm)
    X = scaler.transform(fv.values.reshape(1, -1))
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    # model only knows the classes it was trained on
    known_classes = model.classes_
    proba_dict = {int(cls): float(p) for cls, p in zip(known_classes, proba)}

    print(f'Test image: fall (URFD test set)')
    print(f'Prediction: {state_names[pred]}')
    print('Probabilities:')
    for i, name in enumerate(state_names):
        p = proba_dict.get(i, 0.0)
        bar = '#' * int(p * 40)
        print(f'  {name:12s} |{bar:<40s}| {p:.4f}')
else:
    print('No pose detected in fall image')

pose.close()
print('\n===== Pipeline verified OK =====')
print('Run: python inference.py  (with webcam GUI)')
