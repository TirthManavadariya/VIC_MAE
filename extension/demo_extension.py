"""
demo_extension.py
-----------------
Interactive Web Demo for ViC-MAE Temporal Extension:
Upload an MP4/AVI video, pick a time-gap Delta-t, and visualize:
  1. Context Frame (Frame-i)
  2. Ground Truth Target Frame (Frame-j)
  3. 95% Masked Target Frame
  4. Reconstructed Target Frame (predicted by SiamMAE CrossFrameDecoder)

Usage:
    python demo_extension.py
    Open http://127.0.0.1:5001 in your browser.
"""

from __future__ import annotations

import os
import sys
import io
import base64
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from flask import Flask, request, render_template_string

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Ensure parent directory and ViC-MAE are importable
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "output").exists() else _HERE
_REPO = _HERE / "ViC-MAE" if (_HERE / "ViC-MAE").exists() else _ROOT / "ViC-MAE"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import ExtendedViCMAE
from datasets.video_image_dataset import default_frame_transform

# Paths
CKPT_DIR = _ROOT / "output" / "vicmae_extended"
BASE_CKPT = _ROOT / "vicmae_pretrain_vit_base_in1k_k400.pth"

# ---------------------------------------------------------------------------
# Helpers for patch unpatchification & visual masking
# ---------------------------------------------------------------------------

def unpatchify(x: torch.Tensor, patch_size: int = 16, in_chans: int = 3, img_size: int = 224) -> torch.Tensor:
    """
    x: (B, L, patch_size**2 * in_chans)
    Returns: (B, in_chans, img_size, img_size)
    """
    h = w = img_size // patch_size
    B = x.shape[0]
    p = patch_size
    x = x.reshape(B, h, w, p, p, in_chans)
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(B, in_chans, h * p, w * p)
    return imgs


def tensor_to_b64(tensor: torch.Tensor) -> str:
    """
    Convert (C, H, W) normalized image tensor to base64 JPEG string.
    """
    # Denormalize ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    t = tensor * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    t = (t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(t)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_masked_view(img_tensor: torch.Tensor, mask: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Apply binary mask (1 = masked, 0 = visible) by greying out masked patches.
    """
    # img_tensor: (1, 3, 224, 224)
    # mask: (1, 196)
    B, C, H, W = img_tensor.shape
    p = patch_size
    h, w = H // p, W // p
    mask_2d = mask.reshape(B, 1, h, 1, w, 1).expand(B, C, h, p, w, p).reshape(B, C, H, W)
    masked_img = img_tensor.clone()
    # Replace masked areas with dark placeholder
    masked_img[mask_2d == 1.0] = -0.5
    return masked_img


# ---------------------------------------------------------------------------
# Load Extended Model
# ---------------------------------------------------------------------------

def load_extended_model():
    print("Loading ExtendedViCMAE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExtendedViCMAE(
        decoder_dim=512,
        decoder_depth=4,
        decoder_heads=8,
        img_size=224,
        cross_mask_ratio=0.95,
    )
    # Search for latest checkpoint in output dir
    ckpt_file = None
    if CKPT_DIR.exists():
        candidates = sorted(CKPT_DIR.glob("checkpoint_epoch*.pth"))
        if candidates:
            ckpt_file = candidates[-1]

    if ckpt_file and ckpt_file.exists():
        print(f"Loading extended checkpoint: {ckpt_file}")
        state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=False)
    elif BASE_CKPT.exists():
        print(f"Loading base ViC-MAE weights from {BASE_CKPT}")
        state = torch.load(str(BASE_CKPT), map_location="cpu", weights_only=False)
        model.vicmae.load_state_dict(state.get("model", state), strict=False)
    else:
        print("Notice: No trained checkpoint found. Initialized model with base weights for interactive demonstration.")

    model.eval()
    model.to(device)
    print(f"Extended model loaded on {device}")
    return model, device


# ---------------------------------------------------------------------------
# Web UI HTML Template
# ---------------------------------------------------------------------------

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViC-MAE · Temporal Video Extension</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:        #15120E;
            --surface:   #1E1A14;
            --accent:    #C1443B;
            --accent-hover: #D65248;
            --amber:     #E8A33D;
            --green:     #4EAA78;
            --border:    #352E25;
            --text:      #EDE6DA;
            --muted:     #A89F94;
            --sans:      'Space Grotesk', system-ui, sans-serif;
            --mono:      'JetBrains Mono', 'Consolas', monospace;
        }
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--sans);
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 50px 20px;
        }
        .container { max-width: 900px; width: 100%; }
        .header { margin-bottom: 30px; }
        .header h1 {
            font-size: 1.8em;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #FAF6F0;
        }
        .header h1 .icon { color: var(--accent); }
        .subtitle { color: var(--muted); font-size: 0.95em; margin-top: 6px; }
        .badges { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
        .badge {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 5px 12px;
            font-family: var(--mono);
            font-size: 0.76em;
            letter-spacing: 0.05em;
            color: var(--muted);
            border-radius: 3px;
        }
        .badge span { color: var(--amber); font-weight: 600; }

        .dropzone-wrapper {
            position: relative;
            background: var(--surface);
            border: 1px dashed var(--border);
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .dropzone-wrapper:hover {
            border-color: var(--amber);
            background: #231E17;
        }
        .dropzone-wrapper input { display: none; }
        .drop-title { font-size: 1.1em; font-weight: 600; margin-bottom: 6px; }
        .drop-hint { color: var(--muted); font-size: 0.85em; font-family: var(--mono); }

        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
            background: var(--surface);
            padding: 16px;
            border: 1px solid var(--border);
            border-radius: 4px;
        }
        .control-group label {
            display: block;
            font-family: var(--mono);
            font-size: 0.8em;
            color: var(--muted);
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .control-group input[type="range"] {
            width: 100%;
            accent-color: var(--accent);
        }
        .control-val {
            font-family: var(--mono);
            font-size: 0.82em;
            color: var(--amber);
            float: right;
        }

        .btn {
            display: block;
            width: 100%;
            padding: 14px;
            background: var(--accent);
            color: var(--text);
            border: none;
            font-family: var(--sans);
            font-size: 0.95em;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            cursor: pointer;
            border-radius: 3px;
            transition: background 0.15s;
        }
        .btn:hover { background: var(--accent-hover); }

        .visual-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 30px;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 4px;
            text-align: center;
        }
        .card-title {
            font-family: var(--mono);
            font-size: 0.75em;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 10px;
        }
        .card img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            border: 1px solid #332B21;
            display: block;
            border-radius: 2px;
        }
        .card-tag {
            margin-top: 8px;
            font-family: var(--mono);
            font-size: 0.7em;
            color: var(--amber);
        }

        .metrics-bar {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            background: #191510;
            padding: 12px 18px;
            border: 1px solid var(--border);
            font-family: var(--mono);
            font-size: 0.8em;
            justify-content: space-around;
            border-radius: 4px;
        }
        .metrics-bar span b { color: var(--green); }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 0.72em;
            font-family: var(--mono);
            color: var(--border);
            letter-spacing: 0.06em;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1><span class="icon">⧉</span> ViC-MAE Temporal Extension</h1>
        <p class="subtitle">SiamMAE-style Cross-Frame Reconstruction Conditioned on Continuous Time Gap &Delta;t</p>
        <div class="badges">
            <div class="badge">MASK RATIO: <span>{{ mask_pct or 95 }}%</span></div>
            <div class="badge">TIME GAP &Delta;t: <span>{{ delta_t_val or '0.31' }}</span></div>
            <div class="badge">DECODER: <span>4-LAYER CROSS-ATTN</span></div>
        </div>
    </div>

    <form method="POST" enctype="multipart/form-data" id="upload-form">
        <div class="dropzone-wrapper" onclick="document.getElementById('video-input').click()">
            <div class="drop-title">📁 {{ filename or 'Click to select an MP4 / AVI / MOV video' }}</div>
            <div class="drop-hint">Upload any video clip to see cross-frame future prediction</div>
            <input type="file" name="video" id="video-input" accept="video/*,.mp4,.avi,.mov,.mkv,.webm" onchange="document.getElementById('upload-form').submit()">
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Frame Gap &Delta;t (Frames) <span class="control-val" id="gap-val">{{ frame_gap or 15 }} frames</span></label>
                <input type="range" name="frame_gap" min="1" max="48" value="{{ frame_gap or 15 }}" oninput="document.getElementById('gap-val').innerText = this.value + ' frames'">
            </div>
            <div class="control-group">
                <label>Target Mask Ratio (%) <span class="control-val" id="mask-val">{{ mask_pct or 95 }}%</span></label>
                <input type="range" name="mask_ratio" min="50" max="98" value="{{ mask_pct or 95 }}" oninput="document.getElementById('mask-val').innerText = this.value + '%'">
            </div>
        </div>

        <button type="submit" class="btn">⚡ Run Cross-Frame Prediction</button>
    </form>

    {% if frame_i_b64 %}
    <div class="metrics-bar">
        <span>Context: <b>Frame {{ idx_i }}</b></span>
        <span>Target: <b>Frame {{ idx_j }}</b></span>
        <span>&Delta;t: <b>{{ delta_t_val }}</b></span>
        <span>Masked Patches: <b>186 / 196 (95%)</b></span>
        <span>Loss: <b>{{ loss_val }}</b></span>
    </div>

    <div class="visual-grid">
        <div class="card">
            <div class="card-title">1. Context Frame (i)</div>
            <img src="data:image/jpeg;base64,{{ frame_i_b64 }}" alt="Frame i">
            <div class="card-tag">Full Encoder View</div>
        </div>
        <div class="card">
            <div class="card-title">2. Target Frame (j)</div>
            <img src="data:image/jpeg;base64,{{ frame_j_b64 }}" alt="Frame j Ground Truth">
            <div class="card-tag">Ground Truth Reality</div>
        </div>
        <div class="card">
            <div class="card-title">3. Masked Query (95%)</div>
            <img src="data:image/jpeg;base64,{{ frame_masked_b64 }}" alt="Masked Frame j">
            <div class="card-tag">Decoder Input</div>
        </div>
        <div class="card">
            <div class="card-title">4. Model Prediction</div>
            <img src="data:image/jpeg;base64,{{ frame_pred_b64 }}" alt="Reconstructed Frame j">
            <div class="card-tag">Predicted via SiamMAE</div>
        </div>
    </div>
    {% endif %}

    <div class="footer">ViC-MAE TEMPORAL EXTENSION · ECCV 2024 / SIAM-MAE REPRODUCTION</div>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__)
model, device = None, None
transform = default_frame_transform(224)

@app.route("/", methods=["GET", "POST"])
def index():
    global model, device
    frame_i_b64 = None
    frame_j_b64 = None
    frame_masked_b64 = None
    frame_pred_b64 = None
    loss_val = None
    delta_t_val = None
    idx_i, idx_j = 0, 15
    filename = None

    frame_gap = int(request.form.get("frame_gap", 15))
    mask_pct = int(request.form.get("mask_ratio", 95))
    mask_ratio = mask_pct / 100.0

    if request.method == "POST" and "video" in request.files:
        file = request.files["video"]
        if file.filename:
            filename = file.filename
            ext = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            if _HAS_CV2:
                cap = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 1:
                    actual_gap = min(frame_gap, total - 1)
                    idx_i = 0
                    idx_j = idx_i + actual_gap

                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_i)
                    _, f_i = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_j)
                    _, f_j = cap.read()
                    cap.release()

                    if f_i is not None and f_j is not None:
                        pil_i = Image.fromarray(cv2.cvtColor(f_i, cv2.COLOR_BGR2RGB))
                        pil_j = Image.fromarray(cv2.cvtColor(f_j, cv2.COLOR_BGR2RGB))

                        t_i = transform(pil_i).unsqueeze(0).to(device)
                        t_j = transform(pil_j).unsqueeze(0).to(device)

                        dt = torch.tensor([[actual_gap / 48.0]], device=device, dtype=torch.float32)
                        delta_t_val = f"{actual_gap / 48.0:.2f}"

                        with torch.no_grad():
                            ctx = model.encode_full(t_i)
                            loss, pred, mask = model.forward_cross_recon(
                                ctx_tokens=ctx,
                                target_imgs=t_j,
                                delta_t=dt,
                                mask_ratio=mask_ratio,
                            )
                            # Unpatchify predicted patches
                            pred_imgs = unpatchify(pred, patch_size=16, in_chans=3, img_size=224)
                            masked_imgs = create_masked_view(t_j, mask, patch_size=16)

                            frame_i_b64 = tensor_to_b64(t_i[0])
                            frame_j_b64 = tensor_to_b64(t_j[0])
                            frame_masked_b64 = tensor_to_b64(masked_imgs[0])
                            frame_pred_b64 = tensor_to_b64(pred_imgs[0])
                            loss_val = f"{loss.item():.4f}"

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return render_template_string(
        HTML,
        frame_i_b64=frame_i_b64,
        frame_j_b64=frame_j_b64,
        frame_masked_b64=frame_masked_b64,
        frame_pred_b64=frame_pred_b64,
        loss_val=loss_val,
        delta_t_val=delta_t_val,
        idx_i=idx_i,
        idx_j=idx_j,
        frame_gap=frame_gap,
        mask_pct=mask_pct,
        filename=filename,
    )


if __name__ == "__main__":
    model, device = load_extended_model()
    print("\n" + "=" * 60)
    print("  ViC-MAE Temporal Extension Demo running on http://127.0.0.1:5001")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5001, debug=False)
