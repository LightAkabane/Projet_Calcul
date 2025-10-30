# yolo_dynamic.py — export YOLOv8n ONNX dynamique (H,W dynamiques, sans NMS)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # ou ton .pt custom

# dynamic=True => axes batch/H/W dynamiques
# half=False (FP32) pour WebGPU
# simplify=True pour nettoyer le graphe
# opset 12 (très compatible WebGPU)
model.export(
    format='onnx',
    opset=12,
    dynamic=True,     # <<< IMPORTANT
    simplify=True,
    imgsz=(640, 640),
    half=False
)
print('✅ Export dynamique terminé')
