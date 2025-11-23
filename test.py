import torch
import torchreid
import os

# === CONFIG À ADAPTER ===
PTH_PATH = "osnet_x0_75.pth"         # ton fichier .pth téléchargé
ONNX_PATH = "osnet_x0_5_reid.onnx"  # sortie
HEIGHT = 256
WIDTH = 128

def main():
    # 1) Créer le modèle OSNet x0_5
    model = torchreid.models.build_model(
        name='osnet_x0_5',
        num_classes=1,      # pas important pour l'export features
        pretrained=False
    )

    # 2) Charger les poids
    state = torch.load(PTH_PATH, map_location='cpu')
    # Certains checkpoints ont la clé "state_dict", d'autres non :
    if "state_dict" in state:
        state = state["state_dict"]

    # Enlever éventuellement le "module." si ça vient d'un DataParallel
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_key] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    print("Poids chargés avec succès.")

    # 3) Input factice (1,3,256,128)
    dummy = torch.randn(1, 3, HEIGHT, WIDTH, device='cpu')

    # 4) Export ONNX
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=12
    )

    print(f"Modèle exporté en ONNX : {ONNX_PATH}")

if __name__ == "__main__":
    main()
