import os
import json
from pathlib import Path
import torch
import torchvision.models as models

# Default paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_chili.pth"
DEFAULT_CLASSES_PATH = BASE_DIR / "utils" / "class_names.json"


def resolve_model_path(path: str | Path | None = None) -> Path:
    """Mencari lokasi file model .pth yang valid."""
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
        if (BASE_DIR / path).exists():
            return BASE_DIR / path

    # Fallback paths
    candidates = [
        DEFAULT_MODEL_PATH,
        BASE_DIR / "model" / "mobilenetv2_chili.pth",
        Path("models/mobilenetv2_chili.pth"),
        Path("model/mobilenetv2_chili.pth"),
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"File model tidak ditemukan di {DEFAULT_MODEL_PATH}")


def load_class_names(path: str | Path | None = None) -> list[str]:
    """Membaca daftar nama kelas dari class_names.json."""
    target_path = Path(path) if path else DEFAULT_CLASSES_PATH
    if not target_path.exists():
        target_path = BASE_DIR / "utils" / "class_names.json"
        
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # Default fallback bila file json tidak ditemukan
    return [
        "Bacterial Spot",
        "Cercospora Leaf Spot",
        "Curl Virus",
        "Healthy Leaf",
        "Nutrition Deficiency",
        "White spot"
    ]


def load_model(
    model_path: str | Path | None = None,
    num_classes: int | None = None,
    device: str = "cpu"
):
    """
    Memuat model MobileNetV2 dari file checkpoint .pth secara akurat.
    Mendukung format checkpoint dictionary lengkap (dengan metadata: state_dict,
    class_names, img_size, normalize, dropout) maupun state_dict murni.
    
    Returns:
        model: torch.nn.Module yang siap evaluasi / inferensi
        class_names: list[str] urutan kelas model
        metadata: dict informasi konfigurasi model (img_size, normalize, dropout, dll.)
    """
    actual_path = resolve_model_path(model_path)
    checkpoint = torch.load(actual_path, map_location=device)
    
    metadata = {
        "img_size": 224,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "dropout": 0.2,
        "arch": "mobilenet_v2",
        "model_path": str(actual_path)
    }
    
    # Deteksi struktur checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        class_names = checkpoint.get("class_names")
        if "img_size" in checkpoint:
            metadata["img_size"] = checkpoint["img_size"]
        if "normalize" in checkpoint:
            metadata["normalize"] = checkpoint["normalize"]
        if "dropout" in checkpoint:
            metadata["dropout"] = checkpoint["dropout"]
        if "arch" in checkpoint:
            metadata["arch"] = checkpoint["arch"]
    else:
        state_dict = checkpoint
        class_names = None

    # Tentukan class names
    if class_names is None or len(class_names) == 0:
        class_names = load_class_names()

    actual_num_classes = num_classes if num_classes is not None else len(class_names)
    metadata["class_names"] = class_names
    metadata["num_classes"] = actual_num_classes

    # Inisialisasi MobileNetV2
    model = models.mobilenet_v2(weights=None)
    
    # Pasang dropout rate sesuai checkpoint training
    dropout_rate = metadata.get("dropout", 0.2)
    model.classifier[0] = torch.nn.Dropout(p=dropout_rate)
    
    # Sesuaikan output layer dengan jumlah kelas
    model.classifier[1] = torch.nn.Linear(model.last_channel, actual_num_classes)
    
    # Load bobot terlatih
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Hitung total parameter
    metadata["total_params"] = sum(p.numel() for p in model.parameters())
    metadata["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, class_names, metadata
