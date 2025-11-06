from ultralytics import YOLO

# Charger le modèle léger (ou ton modèle custom)
model = YOLO('yolov8n.pt')

# Exporter vers ONNX
model.export(format='onnx', opset=12)
