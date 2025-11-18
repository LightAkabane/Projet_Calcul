import torch
import torchreid

# 1. Charger un OSNet léger pré-entraîné
model = torchreid.models.build_model(
    name='osnet_x0_25',   # version light
    num_classes=1000,
    pretrained=True       # charge des poids pré-entraînés ReID
)

# On veut le modèle en mode "features", pas en classification
model.classifier = torch.nn.Identity()
model.eval()

# 2. Input factice (B, C, H, W) = (1, 3, 256, 128)
dummy = torch.randn(1, 3, 256, 128)

# 3. Export ONNX
torch.onnx.export(
    model,
    dummy,
    "osnet_reid.onnx",
    input_names=["input"],
    output_names=["feat"],
    opset_version=12,
    do_constant_folding=True
)

print("Export OK -> osnet_reid.onnx")
