import time
import torch
import torch.nn.functional as F


def predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_names: list[str],
    device: str = "cpu"
) -> tuple[dict, int]:
    """
    Melakukan inferensi klasifikasi penyakit daun cabai menggunakan model MobileNetV2.
    Mengekstrak skor logit mentah, energi bebas (Free Energy Score), dan margin logit
    untuk mendeteksi apakah sampel benar-benar berada dalam domain tanaman cabai (In-Distribution)
    atau merupakan daun tanaman lain / objek luar (Out-of-Distribution).
    
    Args:
        model: MobileNetV2 PyTorch model
        input_tensor: Tensor input pra-proses (1, 3, H, W)
        class_names: List nama-nama kelas penyakit
        device: 'cuda' atau 'cpu'
        
    Returns:
        results: Dictionary ringkasan hasil prediksi, metrik domain cabai, dan peringkat probabilitas
        top_idx: Indeks integer kelas teratas
    """
    model.eval()
    tensor_dev = input_tensor.to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor_dev)
        probs = F.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000

    # Metrik Logit & Free Energy untuk OOD Detection
    max_logit = float(torch.max(logits).item())
    energy_score = float(-torch.logsumexp(logits, dim=1).item())
    
    sorted_logits, _ = torch.sort(logits[0], descending=True)
    logit_margin = float((sorted_logits[0] - sorted_logits[1]).item()) if len(sorted_logits) > 1 else max_logit

    # Daun cabai in-distribution memiliki max_logit >= 6.5 dan energy <= -6.8
    is_chili_domain = bool(max_logit >= 6.5 and energy_score <= -6.8)

    conf, idx = torch.max(probs, dim=0)
    top_idx = idx.item()
    top_class_name = class_names[top_idx]

    # Ambil seluruh ranking probabilitas
    sorted_indices = torch.argsort(probs, descending=True).tolist()
    all_predictions = [
        {
            "rank": rank + 1,
            "index": i,
            "class": class_names[i],
            "prob": float(probs[i].item()),
            "pct": float(probs[i].item() * 100)
        }
        for rank, i in enumerate(sorted_indices)
    ]

    top3 = all_predictions[:3]

    results = {
        "class_name": top_class_name,
        "confidence": float(conf.item()),
        "confidence_pct": float(conf.item() * 100),
        "prediction_index": top_idx,
        "latency_ms": float(latency_ms),
        "is_healthy": (top_class_name.lower().strip() == "healthy leaf"),
        "max_logit": max_logit,
        "energy_score": energy_score,
        "logit_margin": logit_margin,
        "is_chili_domain": is_chili_domain,
        "top3": top3,
        "all_predictions": all_predictions,
        "raw_probs": probs.cpu().numpy().tolist()
    }

    return results, top_idx
