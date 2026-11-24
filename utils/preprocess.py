from PIL import Image
import torchvision.transforms as T
import torch

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def get_transform(img_size: int = 224, normalize_dict: dict | None = None) -> T.Compose:
    """
    Membuat PyTorch transform pipeline sesuai spesifikasi pelatihan model.
    """
    mean = DEFAULT_MEAN
    std = DEFAULT_STD
    
    if normalize_dict and "mean" in normalize_dict and "std" in normalize_dict:
        mean = normalize_dict["mean"]
        std = normalize_dict["std"]

    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_and_preprocess(
    image: Image.Image,
    img_size: int = 224,
    normalize_dict: dict | None = None
) -> torch.Tensor:
    """
    Melakukan pra-pemrosesan gambar daun cabai sebelum diinputkan ke model.
    
    Args:
        image: PIL Image masukan dari user
        img_size: Ukuran resolusi target (default: 224)
        normalize_dict: Parameter normalisasi mean & std

    Returns:
        torch.Tensor dengan dimensi (1, 3, img_size, img_size)
    """
    rgb_image = image.convert("RGB")
    transform = get_transform(img_size=img_size, normalize_dict=normalize_dict)
    tensor = transform(rgb_image).unsqueeze(0)
    return tensor
