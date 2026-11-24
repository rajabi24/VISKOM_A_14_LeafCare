import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Implementasi Gradient-weighted Class Activation Mapping (Grad-CAM)
    untuk visualisasi area atensi Convolutional Neural Network (MobileNetV2).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Bersihkan hooks lama bila ada
        self.remove_hooks()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """
        Menghasilkan heatmap aktivasi Grad-CAM ternormalisasi [0, 1].
        
        Args:
            input_tensor: Tensor input berukuran (1, 3, H, W)
            class_idx: Indeks kelas target (bila None, gunakan prediksi argmax)

        Returns:
            cam: 2D numpy array (H_feat, W_feat) ternormalisasi 0.0 - 1.0
            class_idx: Indeks kelas yang divisualisasikan
        """
        self.model.zero_grad()
        
        # Jalankan forward pass dengan autograd aktif
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Target score untuk backward pass
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gagal merekam gradients atau activations dari target layer.")

        gradients = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]     # (C, H, W)

        # Global Average Pooling pada gradien
        weights = gradients.mean(dim=(1, 2))   # (C,)
        
        # Kombinasi linier aktivasi berbobot
        cam = (weights[:, None, None] * activations).sum(dim=0)
        
        # Terapkan ReLU (hanya fitur dengan pengaruh positif terhadap kelas)
        cam = F.relu(cam)
        
        cam_np = cam.detach().cpu().numpy()
        
        # Normalisasi min-max aman
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        denom = cam_max - cam_min
        if denom > 1e-8:
            cam_norm = (cam_np - cam_min) / denom
        else:
            cam_norm = np.zeros_like(cam_np)
            
        return cam_norm, class_idx


def overlay_heatmap(
    original_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "JET"
) -> tuple[Image.Image, Image.Image]:
    """
    Menghasilkan gambar heatmap murni dan gambar overlay perpaduan
    antara foto asli daun dengan Grad-CAM heatmap.
    
    Args:
        original_img: PIL Image asli
        cam: 2D numpy array heatmap [0.0, 1.0]
        alpha: Tingkat transparansi heatmap (0.0 - 1.0)
        colormap: Skema warna ('JET', 'TURBO', 'VIRIDIS', 'INFERNO')

    Returns:
        overlay_pil: PIL Image hasil overlay
        pure_heatmap_pil: PIL Image heatmap murni
    """
    img_rgb = np.array(original_img.convert("RGB"))
    h, w = img_rgb.shape[:2]
    
    # Resize CAM ke ukuran asli gambar dengan interpolasi kubik agar halus
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_clipped = np.clip(cam_resized, 0.0, 1.0)
    cam_uint8 = np.uint8(255 * cam_clipped)
    
    # Pilih colormap OpenCV
    cm_dict = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
        "INFERNO": cv2.COLORMAP_INFERNO if hasattr(cv2, "COLORMAP_INFERNO") else cv2.COLORMAP_JET,
    }
    selected_cm = cm_dict.get(colormap.upper(), cv2.COLORMAP_JET)
    
    heatmap_bgr = cv2.applyColorMap(cam_uint8, selected_cm)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Perpaduan tertimbang (weighted blend)
    alpha = max(0.0, min(1.0, alpha))
    overlay = (alpha * heatmap_rgb + (1.0 - alpha) * img_rgb).astype(np.uint8)
    
    return Image.fromarray(overlay), Image.fromarray(heatmap_rgb)
