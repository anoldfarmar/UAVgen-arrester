
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

COMPOSITE_DIR = Path('/root/workspace/outputs/arrester/stage2/composites_preview_v2')
MANIFEST = COMPOSITE_DIR / 'composite_manifest.json'
OUT = Path('/root/workspace/outputs/arrester/stage2/inpaint_preview_v1')
IMG_DIR = OUT / 'images'
OVERLAY_DIR = OUT / 'overlays'
COMPARE_DIR = OUT / 'compare'
for d in [IMG_DIR, OVERLAY_DIR, COMPARE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = '/root/autodl-tmp/diffusion'
N = 12
STRENGTH = 0.25
STEPS = 25
GUIDANCE = 6.5
PROMPT = 'an aerial inspection image of electrical power equipment, realistic lighting, natural background transition, high fidelity'
NEGATIVE = 'distorted arrester, broken object, duplicated object, blurry, unrealistic shadow, extra object, deformed structure'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

records = json.loads(MANIFEST.read_text())['records'][:N]
print('loading SD2 inpainting:', MODEL_PATH)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    local_files_only=True,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
try:
    pipe.set_progress_bar_config(disable=True)
except Exception:
    pass

def fit_multiple_8(im, is_mask=False):
    w, h = im.size
    nw = max(8, (w // 8) * 8)
    nh = max(8, (h // 8) * 8)
    if (nw, nh) == (w, h):
        return im, (w, h)
    # crop only a few border pixels if needed; avoids rescaling layout geometry.
    return im.crop((0, 0, nw, nh)), (w, h)

outputs = []
thumbs = []
for i, rec in enumerate(records):
    job_id = rec['job_id']
    comp = Image.open(rec['composite_image']).convert('RGB')
    mask = Image.open(rec['inpaint_mask']).convert('L')
    comp_fit, orig_size = fit_multiple_8(comp)
    mask_fit, _ = fit_multiple_8(mask, is_mask=True)
    seed = 20260701 + i
    gen = torch.Generator(device=DEVICE).manual_seed(seed) if DEVICE == 'cuda' else torch.Generator().manual_seed(seed)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=comp_fit,
        mask_image=mask_fit,
        strength=STRENGTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
    ).images[0]
    stem = f'{job_id}_{rec["prototype_id"]}_to_{rec["layout_id"]}'
    out_path = IMG_DIR / f'{stem}.jpg'
    result.save(out_path, quality=95)

    overlay = result.copy().convert('RGB')
    draw = ImageDraw.Draw(overlay)
    tb = rec['target_box_in_context_xyxy']
    pb = rec['pasted_box_xyxy']
    draw.rectangle([float(x) for x in tb], outline=(255, 0, 0), width=3)
    draw.rectangle([float(x) for x in pb], outline=(0, 255, 0), width=3)
    draw.text((8, 8), f'{job_id} strength={STRENGTH}', fill=(255, 255, 0))
    overlay_path = OVERLAY_DIR / f'{stem}.jpg'
    overlay.save(overlay_path, quality=92)

    # side-by-side compare: composite | mask | inpainted overlay
    mask_rgb = Image.merge('RGB', (mask_fit, mask_fit, mask_fit))
    comp_small = comp_fit.copy(); comp_small.thumbnail((360, 280))
    mask_small = mask_rgb.copy(); mask_small.thumbnail((360, 280))
    over_small = overlay.copy(); over_small.thumbnail((360, 280))
    cw = comp_small.width + mask_small.width + over_small.width + 40
    ch = max(comp_small.height, mask_small.height, over_small.height) + 36
    canvas = Image.new('RGB', (cw, ch), 'white')
    x = 10
    canvas.paste(comp_small, (x, 28)); x += comp_small.width + 10
    canvas.paste(mask_small, (x, 28)); x += mask_small.width + 10
    canvas.paste(over_small, (x, 28))
    d = ImageDraw.Draw(canvas)
    d.text((10, 6), f'{job_id}: composite | mask | inpainted', fill=(0, 0, 0))
    compare_path = COMPARE_DIR / f'{stem}.jpg'
    canvas.save(compare_path, quality=92)

    outputs.append({
        **rec,
        'inpainted_image': str(out_path),
        'inpainted_overlay': str(overlay_path),
        'compare': str(compare_path),
        'seed': seed,
        'strength': STRENGTH,
        'steps': STEPS,
        'guidance_scale': GUIDANCE,
        'prompt': PROMPT,
        'negative_prompt': NEGATIVE,
        'original_size': orig_size,
        'processed_size': comp_fit.size,
    })
    t = overlay.copy(); t.thumbnail((360, 260)); thumbs.append((job_id, t))
    print(f'[{i+1:02d}/{len(records)}] {job_id} -> {out_path}')

(OUT / 'inpaint_manifest.json').write_text(json.dumps({'count': len(outputs), 'records': outputs}, ensure_ascii=False, indent=2))
cols = 3
cell_w, cell_h = 400, 310
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
draw = ImageDraw.Draw(sheet)
for i, (jid, thumb) in enumerate(thumbs):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h
    sheet.paste(thumb, (x + 20, y + 34))
    draw.text((x + 10, y + 8), jid, fill=(0, 0, 0))
sheet_path = OUT / 'inpaint_contact_sheet.jpg'
sheet.save(sheet_path, quality=92)
print(json.dumps({'out': str(OUT), 'count': len(outputs), 'contact_sheet': str(sheet_path), 'manifest': str(OUT / 'inpaint_manifest.json')}, ensure_ascii=False, indent=2))
