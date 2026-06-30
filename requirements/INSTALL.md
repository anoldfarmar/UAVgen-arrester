# Diffusion Environment Install Notes

Assume the conda environment is named `diffusion`.

```bash
conda activate diffusion

# 1. Install PyTorch first. Adjust CUDA version if needed.
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Then install the remaining dependencies:

```bash
pip install -r /root/workspace/diffusion-requirement/requirements.txt
pip install -e /root/segment-anything
```

## Verify Imports

```bash
python - <<'PYVERIFY'
import torch
import diffusers
import transformers
import accelerate
import safetensors
import cv2
from segment_anything import sam_model_registry

print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('diffusers:', diffusers.__version__)
print('transformers:', transformers.__version__)
print('ok')
PYVERIFY
```

## Verify Local SD2 Inpainting Model

```bash
python - <<'PYVERIFY'
import torch
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    '/root/autodl-tmp/diffusion',
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,
    requires_safety_checker=False,
)
print(type(pipe).__name__)
print('unet in_channels:', pipe.unet.config.in_channels)
PYVERIFY
```

## Verify SAM Checkpoint

```bash
python - <<'PYVERIFY'
import torch
from segment_anything import sam_model_registry

sam = sam_model_registry['vit_h'](checkpoint='/root/autodl-tmp/sam/sam_vit_h_4b8939.pth')
print('sam loaded')
print('cuda:', torch.cuda.is_available())
PYVERIFY
```
