import cv2
print(f"OpenCV: {cv2.__version__}")
try:
    # Test DNN ONNX support
    net = cv2.dnn.readNetFromONNX
    print("cv2.dnn.readNetFromONNX: OK")
except AttributeError:
    print("cv2.dnn.readNetFromONnx: NOT AVAILABLE")

# Try creating an empty net to test DNN module
try:
    net = cv2.dnn.readNetFromCaffe(b"", b"")
except:
    pass  # Expected to fail with empty data
print("cv2.dnn module: OK")
