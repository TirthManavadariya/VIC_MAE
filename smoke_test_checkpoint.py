"""
Smoke test: load ViC-MAE ViT-B/16 checkpoint and run one forward pass.

Usage:
    python smoke_test_checkpoint.py [--checkpoint PATH] [--repo_dir PATH]
"""
import argparse
from pathlib import Path
import sys
import torch

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Smoke test ViC-MAE ViT-B/16 checkpoint")
parser.add_argument("--checkpoint", type=str, default=str(BASE_DIR / "vicmae_pretrain_vit_base_in1k_k400.pth"), help="Path to checkpoint .pth")
parser.add_argument("--repo_dir", type=str, default=str(BASE_DIR / "ViC-MAE"), help="Path to cloned upstream ViC-MAE repository")
args = parser.parse_args()

CHECKPOINT = Path(args.checkpoint)
REPO_DIR = Path(args.repo_dir)
NUM_CLASSES = 200   # Tiny-ImageNet

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

try:
    import models_vit
except ImportError:
    print(f"Error: Could not import models_vit from {REPO_DIR}. Ensure the upstream ViC-MAE repo is cloned.")
    sys.exit(1)

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Build model
print("\nBuilding vit_base_patch16 ...")
model = models_vit.vit_base_patch16(
    num_classes=NUM_CLASSES,
    drop_path_rate=0.1,
    global_pool=False,
)

# Load checkpoint
print(f"Loading checkpoint: {CHECKPOINT}")
if not CHECKPOINT.exists():
    print(f"Error: Checkpoint file not found at {CHECKPOINT}.")
    print("Please download vicmae_pretrain_vit_base_in1k_k400.pth from HuggingFace and place it in the project root or specify --checkpoint.")
    sys.exit(1)
ckpt = torch.load(CHECKPOINT, map_location="cpu")

# Checkpoint may be a dict with 'model' key (MAE-style)
if "model" in ckpt:
    state_dict = ckpt["model"]
    print(f"  Found 'model' key with {len(state_dict)} tensors")
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
    print(f"  Found 'state_dict' key with {len(state_dict)} tensors")
else:
    state_dict = ckpt
    print(f"  Flat checkpoint with {len(state_dict)} tensors")

print(f"  First 5 keys: {list(state_dict.keys())[:5]}")

# Use the repo's interpolate_pos_embed utility
sys.path.insert(0, REPO_DIR)
from util.pos_embed import interpolate_pos_embed
interpolate_pos_embed(model, state_dict)

# Load — allow missing/unexpected (head will be missing for pretrain ckpt)
msg = model.load_state_dict(state_dict, strict=False)
print(f"\nLoad result:")
print(f"  Missing keys:    {msg.missing_keys}")
print(f"  Unexpected keys: {msg.unexpected_keys[:5]} ...")

# Forward pass on dummy batch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
x = torch.randn(2, 3, 224, 224, device=device)
with torch.no_grad():
    out = model(x)

print(f"\nForward pass OK!")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {out.shape}  (expected [2, {NUM_CLASSES}])")
assert out.shape == (2, NUM_CLASSES), f"Shape mismatch: {out.shape}"
print("\n[SMOKE TEST PASSED]")
