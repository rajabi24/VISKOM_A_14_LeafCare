"""
leaf_validator.py - Modul validasi citra untuk memverifikasi apakah gambar masukan
merupakan daun tanaman (in-distribution) atau objek selain daun (out-of-distribution),
serta mengevaluasi tingkat keyakinan (confidence) inferensi model.
"""

import cv2
import numpy as np
from PIL import Image


def validate_leaf_presence(
    image: Image.Image,
    min_foliage_ratio: float = 0.08,
    min_green_ratio: float = 0.03,
    min_texture_std: float = 10.0
) -> tuple[bool, str, str, dict]:
    """
    Memvalidasi apakah gambar yang diunggah memiliki karakteristik helai daun tanaman
    berdasarkan spektrum warna klorofil/jaringan daun (HSV), konsistensi spasial (morfologi),
    dan variasi tekstur alami helai daun.
    Menolak objek non-daun seperti wajah manusia, kendaraan, cangkir, kertas kosong, atau layar biru.

    Args:
        image: PIL Image input
        min_foliage_ratio: Ambang batas minimal total spektrum daun (default 8%)
        min_green_ratio: Ambang batas minimal klorofil hijau (default 3%)
        min_texture_std: Ambang batas variasi piksel untuk mencegah gambar polos

    Returns:
        is_valid: bool (True jika valid daun tanaman, False jika bukan daun)
        status_code: str ('VALID_LEAF', 'NOT_A_LEAF', 'BLANK_IMAGE', 'EXTREME_COLOR')
        reason: str (Penjelasan dalam bahasa Indonesia)
        metrics: dict (Rincian metrik citra untuk transparansi)
    """
    rgb = np.array(image.convert("RGB"))
    h, w, _ = rgb.shape
    total_pixels = h * w

    # 1. Cek Resolusi Minimum
    if h < 50 or w < 50:
        return False, "LOW_RESOLUTION", "Resolusi gambar terlalu kecil (minimal 50×50 piksel).", {
            "height": h, "width": w, "foliage_ratio": 0.0, "green_ratio": 0.0
        }

    # 2. Cek Keseragaman / Gambar Polos (Blank Image Check)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    texture_std = float(np.std(gray))
    if texture_std < min_texture_std:
        return False, "BLANK_IMAGE", "Gambar terlalu polos atau seragam, tidak ditemukan tekstur helai daun.", {
            "texture_std": round(texture_std, 2), "foliage_ratio": 0.0, "green_ratio": 0.0
        }

    # 3. Analisis Spektrum Warna Daun (HSV Color Space)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]

    # Mask klorofil hijau & hijau kekuningan (daun sehat / bercak awal)
    mask_green = (h_chan >= 24) & (h_chan <= 88) & (s_chan >= 22) & (v_chan >= 22)

    # Mask klorosis kuning (defisiensi hara / virus / halo kuning)
    mask_yellow = (h_chan >= 14) & (h_chan <= 32) & (s_chan >= 28) & (v_chan >= 30)

    # Mask nekrosis kecokelatan (bercak cercospora, nekrotik bakteri, daun mengering)
    mask_brown = (h_chan >= 7) & (h_chan <= 23) & (s_chan >= 25) & (v_chan >= 20) & (v_chan <= 190)

    # Filter Morfologi Spasial untuk klorofil hijau
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_green_mask = cv2.morphologyEx(mask_green.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    green_pixels = int(np.sum(clean_green_mask > 0))
    green_ratio = float(green_pixels / total_pixels)

    # Gabungan seluruh spektrum jaringan daun cabai
    raw_foliage_mask = (mask_green | mask_yellow | mask_brown).astype(np.uint8) * 255
    clean_foliage_mask = cv2.morphologyEx(raw_foliage_mask, cv2.MORPH_OPEN, kernel)
    foliage_pixels = int(np.sum(clean_foliage_mask > 0))
    foliage_ratio = float(foliage_pixels / total_pixels)

    # 4. Deteksi Warna Non-Daun yang Sangat Dominan (Alien Colors)
    # Biru / Cyan (langit, pakaian, mobil, layar elektronik)
    mask_blue = (h_chan >= 92) & (h_chan <= 135) & (s_chan >= 35) & (v_chan >= 35)
    blue_ratio = float(np.sum(mask_blue) / total_pixels)

    # Ungu / Magenta / Pink terang
    mask_purple = (h_chan >= 136) & (h_chan <= 170) & (s_chan >= 35) & (v_chan >= 35)
    purple_ratio = float(np.sum(mask_purple) / total_pixels)

    metrics = {
        "foliage_ratio": round(foliage_ratio * 100, 2),
        "green_ratio": round(green_ratio * 100, 2),
        "blue_ratio": round(blue_ratio * 100, 2),
        "purple_ratio": round(purple_ratio * 100, 2),
        "texture_std": round(texture_std, 2),
        "height": h,
        "width": w
    }

    # Aturan Keputusan 1: Klorofil hijau tidak mencukupi (mencegah kayu, kulit, benda non-tanaman)
    if green_ratio < min_green_ratio:
        return False, "NOT_A_LEAF", (
            f"Karakteristik klorofil hijau daun cabai tidak terdeteksi (rasio hijau hanya {green_ratio*100:.1f}%, "
            f"ambang minimal {min_green_ratio*100:.1f}%)."
        ), metrics

    # Aturan Keputusan 2: Total jaringan daun tidak mencukupi
    if foliage_ratio < min_foliage_ratio:
        return False, "NOT_A_LEAF", (
            f"Total spektrum daun cabai tidak mencukupi (area daun {foliage_ratio*100:.1f}%, "
            f"ambang minimal {min_foliage_ratio*100:.0f}%)."
        ), metrics

    # Aturan Keputusan 3: Dominasi warna non-daun ekstrem
    if blue_ratio > 0.35 and foliage_ratio < 0.25:
        return False, "EXTREME_COLOR", (
            f"Gambar didominasi warna biru ({blue_ratio*100:.1f}%) yang bukan karakteristik daun tanaman."
        ), metrics

    if purple_ratio > 0.35 and foliage_ratio < 0.25:
        return False, "EXTREME_COLOR", (
            f"Gambar didominasi warna ungu/magenta ({purple_ratio*100:.1f}%) yang bukan karakteristik daun tanaman."
        ), metrics

    return True, "VALID_LEAF", "Karakteristik fisik dan visual daun tanaman terverifikasi valid.", metrics


def evaluate_prediction_confidence(results: dict, min_confidence: float = 0.50) -> tuple[bool, str]:
    """
    Mengevaluasi apakah tingkat keyakinan (confidence) inferensi model
    mencukupi untuk dianggap sebagai diagnosis yang meyakinkan.
    """
    conf = results.get("confidence", 0.0)
    conf_pct = conf * 100
    
    if conf < min_confidence:
        return False, (
            f"Tingkat keyakinan model rendah ({conf_pct:.1f}% < {min_confidence*100:.0f}%). "
            "Model tidak sepenuhnya yakin dengan diagnosis ini. Disarankan mengambil foto daun lebih dekat dan fokus."
        )
        
    return True, f"Tingkat keyakinan memadai ({conf_pct:.1f}%)."
