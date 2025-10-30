# REID.py — Export OSNet x0.25 -> ONNX WebGPU-friendly
# SANS F.normalize dans le graphe (on normalise côté JS)
# - FP32
# - opset 12
# - batch dynamique
# - simplification onnxsim si dispo

import os, sys, torch, torch.nn as nn
try:
    from torchreid import models
except ImportError:
    print("pip install torchreid"); sys.exit(1)

import onnx
try:
    from onnxsim import simplify
    HAS_ONNXSIM = True
except Exception:
    HAS_ONNXSIM = False

MODEL_NAME = "osnet_x0_25"
H, W = 256, 128
OPSET = 12
OUT_PATH = f"{MODEL_NAME}_msmt17_webgpu_noL2.onnx"

print(f"Chargement {MODEL_NAME} (pretrain ImageNet)…")
backbone = models.osnet_x0_25(num_classes=1000, pretrained=True).eval()

# Wrapper qui retourne le feature vector BRUT (pas de normalisation)
class ReIDWrapper(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m
        if hasattr(self.m, 'classifier') and isinstance(self.m.classifier, nn.Module):
            self.m.classifier = nn.Identity()
    def forward(self, x):
        feat = self.m(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return feat  # <— pas de F.normalize

net = ReIDWrapper(backbone).eval()

dummy = torch.randn(1, 3, H, W)
dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

print(f"Export ONNX -> {OUT_PATH}")
torch.onnx.export(
    net, dummy, OUT_PATH,
    input_names=["input"], output_names=["output"],
    opset_version=OPSET, do_constant_folding=True,
    dynamic_axes=dynamic_axes,
)

print("Vérification ONNX…")
m = onnx.load(OUT_PATH)
onnx.checker.check_model(m)
print("OK ✅")

if HAS_ONNXSIM:
    print("Simplification onnxsim…")
    sm, ok = simplify(m)
    if ok:
        onnx.save(sm, OUT_PATH)
        print("Modèle simplifié ✅")
    else:
        print("Simplify check=False, on garde le modèle d’origine")

# Affiche I/O
m2 = onnx.load(OUT_PATH)
def dims(t):
    return [d.dim_value if d.HasField("dim_value") else "dyn"
            for d in t.type.tensor_type.shape.dim]
for i in m2.graph.input:
    print("input ", i.name, dims(i))
for o in m2.graph.output:
    print("output", o.name, dims(o))

print(f"Terminé 🎉  {os.path.abspath(OUT_PATH)}  ({os.path.getsize(OUT_PATH)/1024:.1f} KB)")
