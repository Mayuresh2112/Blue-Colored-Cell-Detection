from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import uvicorn
import os
import cv2
import time
import psutil
import base64
from ultralytics import YOLO
import torch
import torchvision.ops as ops  # For NMS
import onnxruntime  # For ONNX inference
import numpy as np
import tensorflow as tf  # For TensorFlow Lite inference

app = FastAPI(title="Microscopy Inference Server")

# Paths for different model formats
MODEL_PATHS = {
    "pt": "runs/detect/train/weights/best.pt",
    "onnx": "runs/detect/train/weights/best.onnx",
    "tflite": "runs/detect/train/weights/best.tflite"
}

# Load PyTorch YOLO model at startup for .pt inference
model_pt = YOLO(MODEL_PATHS["pt"])

def slice_image(img, slice_size=1024, overlap=0.2):
    h, w = img.shape[:2]
    stride = int(slice_size * (1 - overlap))
    slices = []
    coordinates = []
    for y in range(0, h, stride):
        if y + slice_size > h:
            y = h - slice_size
        for x in range(0, w, stride):
            if x + slice_size > w:
                x = w - slice_size
            slice_img = img[y:y+slice_size, x:x+slice_size]
            slices.append(slice_img)
            coordinates.append((x, y))
        if y + slice_size >= h:
            break
    return slices, coordinates

def merge_boxes(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return [], []
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
    merged_boxes = boxes_tensor[keep].tolist()
    merged_scores = scores_tensor[keep].tolist()
    return merged_boxes, merged_scores

def run_inference_pt(slice_img, conf_threshold, imgsz=1024):
    results = model_pt.predict(source=slice_img, conf=conf_threshold, imgsz=imgsz)
    return results

import time

def run_inference_onnx(slice_img, session, conf_threshold=0.25, input_shape=(1, 3, 1024, 1024)):
    t0 = time.time()

    # Preprocess
    img_resized = cv2.resize(slice_img, (input_shape[3], input_shape[2]))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_expanded = np.expand_dims(img_transposed, axis=0)

    preprocess_time = time.time() - t0
    t1 = time.time()

    # Inference
    inputs = {session.get_inputs()[0].name: img_expanded}
    outputs = session.run(None, inputs)
    preds = outputs[0]
    infer_time = time.time() - t1
    t2 = time.time()

    # Postprocess
    preds = np.squeeze(preds, 0)
    boxes, scores = [], []
    for pred in preds:
        x, y, w, h = pred[0:4]
        obj_conf = pred[4]
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        conf = obj_conf * class_scores[class_id]
        if conf > conf_threshold:
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

    postprocess_time = time.time() - t2

    # Print logs in YOLO style
    print(f"ONNX: {len(boxes)} detections, "
          f"{preprocess_time*1000:.1f}ms preprocess, "
          f"{infer_time*1000:.1f}ms inference, "
          f"{postprocess_time*1000:.1f}ms postprocess "
          f"per image at shape {img_expanded.shape}")

    return boxes, scores



def run_inference_tflite(slice_img, interpreter, score_threshold=0.3):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    h, w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    img_resized = cv2.resize(slice_img, (w, h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract outputs (typical SSD/YOLO-style TFLite models)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]     # [N,4]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0] # [N]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]    # [N]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    # Convert normalized box coordinates back to image size
    h_orig, w_orig, _ = slice_img.shape
    final_boxes, final_scores = [], []
    for i in range(num_detections):
        score = scores[i]
        if score >= score_threshold:
            # TFLite boxes are [ymin, xmin, ymax, xmax] normalized [0,1]
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w_orig), int(ymin * h_orig)
            x2, y2 = int(xmax * w_orig), int(ymax * h_orig)
            final_boxes.append([x1, y1, x2, y2])
            final_scores.append(float(score))

    return final_boxes, final_scores

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head>
            <title>Microscopy Inference Server</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f8f9fa; text-align: center; padding: 40px; }
                h2 { color: #2c3e50; }
                form { background: white; padding: 20px; border-radius: 10px; 
                       box-shadow: 0 4px 10px rgba(0,0,0,0.1); display: inline-block; }
                input, select { padding: 8px; margin: 10px; width: 250px; border-radius: 5px; border: 1px solid #ccc; }
                input[type="submit"] { background: #2ecc71; color: white; font-weight: bold; cursor: pointer; border: none; }
                input[type="submit"]:hover { background: #27ae60; }
            </style>
        </head>
        <body>
            <h2>🔬 Microscopy Inference Server</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input type="file" name="file" required><br>
                <label for="model_format">Model Format:</label><br>
                <select name="model_format">
                    <option value="pt">PyTorch (.pt)</option>
                    <option value="onnx">ONNX (.onnx)</option>
                    <option value="tflite">TensorFlow Lite (.tflite)</option>
                </select><br>
                <label for="conf_threshold">Confidence Threshold:</label><br>
                <input type="text" name="conf_threshold" value="0.25"><br>
                <input type="submit" value="Run Inference">
            </form>
        </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25)
):
    input_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(input_path)
    slice_size = 1024
    overlap = 0.2
    slices, coords = slice_image(img, slice_size, overlap)

    formats = ["pt", "onnx", "tflite"]
    results_table = "<table border='1' style='margin:auto;'><tr><th>Format</th><th>Time (s)</th><th>Memory (MB)</th><th>Detections</th></tr>"
    annotated_imgs = {}

    for fmt in formats:
        all_boxes = []
        all_scores = []
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        if fmt == "onnx" and os.path.exists(MODEL_PATHS["onnx"]):
            session = onnxruntime.InferenceSession(MODEL_PATHS["onnx"])
        elif fmt == "tflite" and os.path.exists(MODEL_PATHS["tflite"]):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATHS["tflite"])
            interpreter.allocate_tensors()

        for slice_img, (x_off, y_off) in zip(slices, coords):
            if fmt == "pt" and os.path.exists(MODEL_PATHS["pt"]):
                results = run_inference_pt(slice_img, conf_threshold, imgsz=slice_size)
                for result in results:
                    if len(result.boxes) == 0:
                        continue
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    for box, score in zip(boxes_xyxy, scores):
                        x1, y1, x2, y2 = box
                        shifted_box = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off]
                        all_boxes.append(shifted_box)
                        all_scores.append(score)
            elif fmt == "onnx" and os.path.exists(MODEL_PATHS["onnx"]):
                boxes, scores = run_inference_onnx(slice_img, session)
                boxes = [[x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off] for x1, y1, x2, y2 in boxes]
                all_boxes.extend(boxes)
                all_scores.extend(scores)
            elif fmt == "tflite" and os.path.exists(MODEL_PATHS["tflite"]):
                boxes, scores = run_inference_tflite(slice_img, interpreter)
                boxes = [[x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off] for x1, y1, x2, y2 in boxes]
                all_boxes.extend(boxes)
                all_scores.extend(scores)

        merged_boxes, merged_scores = merge_boxes(all_boxes, all_scores, iou_threshold=0.5)

        img_copy = img.copy()
        for box, score in zip(merged_boxes, merged_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{score:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        output_path = f"outputs/annotated_{fmt}_{file.filename}"
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(output_path, img_copy)
        annotated_imgs[fmt] = output_path

        inference_time = round(time.time() - start_time, 3)
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = round(mem_after - mem_before, 2)
        results_table += f"<tr><td>{fmt}</td><td>{inference_time}</td><td>{mem_used}</td><td>{len(merged_boxes)}</td></tr>"

    results_table += "</table>"

    # Show annotated image for PyTorch by default
    show_fmt = "pt" if "pt" in annotated_imgs else list(annotated_imgs.keys())[0]
    annotated_img = cv2.imread(annotated_imgs[show_fmt])
    _, buffer = cv2.imencode(".png", annotated_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return f"""
    <html>
        <head>
            <title>Microscopy Inference Server</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #f8f9fa; text-align: center; padding: 40px; }}
                h2 {{ color: #2c3e50; }}
                .card {{ background: white; padding: 20px; border-radius: 10px; 
                         box-shadow: 0 4px 10px rgba(0,0,0,0.1); display: inline-block; }}
                .metrics {{ margin-top: 20px; text-align: left; }}
                img {{ margin-top: 20px; border-radius: 10px; max-width: 90%; height: auto; }}
                a {{ display: inline-block; margin-top: 10px; text-decoration: none; color: white; 
                     background: #3498db; padding: 10px 20px; border-radius: 5px; }}
                a:hover {{ background: #2980b9; }}
                table {{ margin-top: 20px; border-collapse: collapse; }}
                th, td {{ padding: 8px 12px; }}
            </style>
        </head>
        <body>
            <h2>🔬 Detection Result</h2>
            <div class="card">
                <p><b>📂 File:</b> {file.filename}</p>
                <img src="data:image/png;base64,{img_base64}">
                <br>
                {results_table}
                <a href="/">⬅️ Upload Another Image</a>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
