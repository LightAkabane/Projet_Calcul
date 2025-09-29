from ultralytics import YOLO
m = YOLO("yolov8n.pt")  # ou yolo11n.pt
m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
