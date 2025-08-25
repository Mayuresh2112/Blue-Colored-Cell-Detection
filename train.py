from ultralytics import YOLO

# 1. Load YOLOv11 model
model = YOLO("yolo11n.pt") 

# 2. Train (uses train + val from dataset.yaml)
model.train(
    data="C:/Users/mayur/Downloads/Task/Task/dataset.yaml",
    imgsz=1024,
    epochs=50,
    batch=4,     # smaller batch for CPU
    device="cpu" # force CPU
)


# 3. Evaluate on Validation Set
metrics_val = model.val(device="cpu")
print("Validation metrics:", metrics_val)

# 4. Evaluate on Test Set
metrics_test = model.val(
    data="C:/Users/mayur/Downloads/Task/Task/dataset.yaml",
    split="test",
    device="cpu"
)
print("Test metrics:", metrics_test)


# 5. Best model path
print("Best model is saved at: runs/detect/train/weights/best.pt")

