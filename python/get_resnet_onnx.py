import torch
import torchreid

model = torchreid.models.build_model(
    name='resnet18',
    num_classes=1000,
    pretrained=True
)

# On supprime la tête de classification
model.fc = torch.nn.Identity()
model.eval()

dummy = torch.randn(1, 3, 256, 128)

torch.onnx.export(
    model,
    dummy,
    "resnet18_reid.onnx",
    input_names=["input"],
    output_names=["feat"],
    opset_version=16,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "feat": {0: "batch"}}
)
