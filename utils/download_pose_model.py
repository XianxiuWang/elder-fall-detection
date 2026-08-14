#!/usr/bin/env python3
"""Download MediaPipe Pose Landmarker model"""
import urllib.request
import sys

urls = [
    'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task',
]

for url in urls:
    try:
        print(f"Trying: {url}")
        urllib.request.urlretrieve(url, r'E:\老人跌倒\pose_landmarker.task')
        print(f"SUCCESS!")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL: {e}")

print("All URLs failed!")
sys.exit(1)
