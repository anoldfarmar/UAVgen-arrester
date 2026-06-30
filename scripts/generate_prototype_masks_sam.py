
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from segment_anything import sam_model_registry, SamPredictor

STAGE1 = Path('/root/workspace/outputs/arrester/stage1')
BANK_PATH = Path('/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json')
OUT = Path('/root/workspace/outputs/arrester/prototypes/bank')
MASK_DIR = OUT / 'masks'
OVERLAY_DIR = OUT / 'mask_overlays'
CUTOUT_DIR = OUT / 'cutouts_rgba'
REPORT_PATH = OUT / 'mask_quality_report.json'
CONTACT_PATH = OUT / 'mask_contact_sheet.jpg'
SAM_CKPT = Path('/root/autodl-tmp/sam/sam_vit_h_4b8939.pth')
MODEL_TYPE = 'vit_h'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

for d in [MASK_DIR, OVERLAY_DIR, CUTOUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

bank = json.loads(BANK_PATH.read_text())
prototypes = bank['prototypes']

background = {}
with (STAGE1 / 'background_tags_auto.jsonl').open() as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            background[r['crop_id']] = r

print(f'loading SAM {MODEL_TYPE} on {DEVICE}')
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CKPT))
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

records = []
thumbs = []
for idx, proto in enumerate(prototypes):
    pid = proto['prototype_id']
    image_path = Path(proto['image'])
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        records.append({'prototype_id': pid, 'status': 'failed', 'reason': 'image read failed'})
        continue
    h, w = image_bgr.shape[:2]
    bg = background.get(pid)
    if not bg:
        records.append({'prototype_id': pid, 'status': 'failed', 'reason': 'missing focal crop metadata'})
        continue
    bx1, by1, bx2, by2 = bg['bbox_xyxy']
    cx1, cy1, cx2, cy2 = bg['crop_box_xyxy']
    box = np.array([bx1 - cx1, by1 - cy1, bx2 - cx1, by2 - cy1], dtype=np.float32)
    box[0::2] = np.clip(box[0::2], 0, w - 1)
    box[1::2] = np.clip(box[1::2], 0, h - 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(box=box, multimask_output=True)
    # Prefer mask with high score but not absurdly huge. Avoid selecting a whole-tower mask.
    box_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
    best_i = 0
    best_rank = -1e9
    mask_infos = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        area = int(mask.sum())
        area_ratio = area / float(w * h)
        box_area_ratio = area / box_area
        # Score penalty for very large masks relative to bbox.
        penalty = 0.0
        if box_area_ratio > 1.7:
            penalty += (box_area_ratio - 1.7) * 0.25
        if area_ratio > 0.65:
            penalty += 0.5
        rank = float(score) - penalty
        mask_infos.append({'candidate': i, 'score': float(score), 'area_px': area, 'area_ratio': area_ratio, 'mask_to_box_area': box_area_ratio, 'rank': rank})
        if rank > best_rank:
            best_rank = rank
            best_i = i
    mask = masks[best_i].astype(np.uint8) * 255
    score = float(scores[best_i])
    mask_path = MASK_DIR / f'{pid}.png'
    cv2.imwrite(str(mask_path), mask)

    # Overlay
    overlay = image_rgb.copy()
    color = np.zeros_like(overlay)
    color[:, :, 1] = 255
    alpha = (mask > 0).astype(np.float32) * 0.42
    overlay = (overlay * (1 - alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)
    draw_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(draw_img)
    draw.rectangle([float(x) for x in box.tolist()], outline=(255, 0, 0), width=max(2, w // 120))
    draw.text((8, 8), f'{pid} {proto["grade"]} score={score:.3f}', fill=(255, 255, 0))
    overlay_path = OVERLAY_DIR / f'{pid}.jpg'
    draw_img.save(overlay_path, quality=92)

    # RGBA cutout with transparent background
    rgba = np.dstack([image_rgb, mask])
    cutout_path = CUTOUT_DIR / f'{pid}.png'
    Image.fromarray(rgba).save(cutout_path)

    area = int((mask > 0).sum())
    xys = np.argwhere(mask > 0)
    if len(xys):
        my1, mx1 = xys.min(axis=0)
        my2, mx2 = xys.max(axis=0)
        mask_bbox = [int(mx1), int(my1), int(mx2), int(my2)]
    else:
        mask_bbox = None
    records.append({
        'prototype_id': pid,
        'grade': proto['grade'],
        'status': 'ok',
        'image': str(image_path),
        'mask': str(mask_path),
        'overlay': str(overlay_path),
        'cutout_rgba': str(cutout_path),
        'sam_score': score,
        'selected_candidate': int(best_i),
        'candidate_masks': mask_infos,
        'box_in_crop_xyxy': [float(x) for x in box.tolist()],
        'mask_area_px': area,
        'mask_area_ratio': area / float(w * h),
        'mask_to_box_area': area / box_area,
        'mask_bbox_xyxy': mask_bbox,
        'quality_flags': []
    })
    rec = records[-1]
    if rec['mask_area_ratio'] < 0.01:
        rec['quality_flags'].append('mask_too_small')
    if rec['mask_to_box_area'] > 1.8:
        rec['quality_flags'].append('mask_much_larger_than_bbox')
    if score < 0.85:
        rec['quality_flags'].append('low_sam_score')

    t = draw_img.copy()
    t.thumbnail((320, 240))
    thumbs.append((pid, proto['grade'], t))
    print(f'[{idx+1:02d}/{len(prototypes)}] {pid}: score={score:.3f} area_ratio={rec["mask_area_ratio"]:.3f} flags={rec["quality_flags"]}')

REPORT_PATH.write_text(json.dumps({'sam_checkpoint': str(SAM_CKPT), 'model_type': MODEL_TYPE, 'device': DEVICE, 'total': len(records), 'records': records}, ensure_ascii=False, indent=2))

# Contact sheet
cols = 4
cell_w, cell_h = 360, 290
rows = int(np.ceil(len(thumbs) / cols))
sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
draw = ImageDraw.Draw(sheet)
for i, (pid, grade, thumb) in enumerate(thumbs):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h
    sheet.paste(thumb, (x + 20, y + 30))
    draw.text((x + 10, y + 8), f'{i+1}. {pid} grade={grade}', fill=(0, 0, 0))
sheet.save(CONTACT_PATH, quality=92)
print(json.dumps({'report': str(REPORT_PATH), 'contact_sheet': str(CONTACT_PATH), 'mask_dir': str(MASK_DIR), 'overlay_dir': str(OVERLAY_DIR), 'cutout_dir': str(CUTOUT_DIR)}, ensure_ascii=False, indent=2))
