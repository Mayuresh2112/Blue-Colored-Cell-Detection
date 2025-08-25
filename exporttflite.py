from ultralytics import YOLO
import onnx
import tf2onnx
import tensorflow as tf

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Export to ONNX format first
model.export(format="onnx", device="cpu")

# Convert ONNX to TensorFlow
onnx_model = onnx.load("runs/detect/train/weights/best.onnx")
tf_rep = tf2onnx.convert.from_onnx(onnx_model)
tf_model = tf_rep[0]

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
