
import json
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

STAGE1 = Path('/root/workspace/outputs/arrester/stage1')
PROTO_BANK = Path('/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json')
MANIFEST = Path('/root/workspace/outputs/arrester/stage2/generation_manifest_v1.json')
OUT = Path('/root/workspace/outputs/arrester/stage2/composites_preview_v1')
IMG_DIR = OUT / 'images'
MASK_DIR = OUT / 'masks_object'
INPAINT_DIR = OUT / 'masks_inpaint_boundary'
OVERLAY_DIR = OUT / 'overlays'
CUTOUT_DIR = Path('/root/workspace/outputs/arrester/prototypes/bank/cutouts_rgba')
MASK_PROTO_DIR = Path('/root/workspace/outputs/arrester/prototypes/bank/masks')

for d in [IMG_DIR, MASK_DIR, INPAINT_DIR, OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N = 40
SEED = 20260630
random.seed(SEED)

manifest = json.loads(MANIFEST.read_text())
jobs = manifest['jobs'][:N]
layout = {x['layout_id']: x for x in json.loads((STAGE1 / 'layout_pool.json').read_text())['items']}
proto_bank = json.loads(PROTO_BANK.read_text())
protos = {p['prototype_id']: p for p in proto_bank['prototypes']}


def feather_alpha(alpha, blur=7):
    if blur <= 0:
        return alpha
    k = blur * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def color_match_rgb(src_rgb, dst_rgb, alpha):
    # Lightweight mean/std match inside object/bbox area. Conservative to avoid damaging real object color.
    mask = alpha > 20
    if mask.sum() < 20:
        return src_rgb
    src = src_rgb.astype(np.float32)
    dst = dst_rgb.astype(np.float32)
    out = src.copy()
    for c in range(3):
        s_vals = src[:, :, c][mask]
        d_vals = dst[:, :, c].reshape(-1)
        sm, ss = s_vals.mean(), s_vals.std() + 1e-6
        dm, ds = d_vals.mean(), d_vals.std() + 1e-6
        matched = (src[:, :, c] - sm) / ss * min(ds, ss * 1.25) + dm
        out[:, :, c] = src[:, :, c] * 0.55 + matched * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_rgba(background_rgb, rgba, target_box, preserve_aspect=True):
    H, W = background_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in target_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    tw, th = max(1, x2 - x1), max(1, y2 - y1)
    src_h, src_w = rgba.shape[:2]
    if preserve_aspect:
        scale = min(tw / src_w, th / src_h)
        nw, nh = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
        px1 = x1 + (tw - nw) // 2
        py1 = y1 + (th - nh) // 2
        px2, py2 = px1 + nw, py1 + nh
    else:
        nw, nh = tw, th
        px1, py1, px2, py2 = x1, y1, x2, y2
    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if nw < src_w else cv2.INTER_CUBIC)
    rgb = resized[:, :, :3]
    alpha = resized[:, :, 3]
    roi = background_rgb[py1:py2, px1:px2].copy()
    rgb = color_match_rgb(rgb, roi, alpha)
    a = feather_alpha(alpha.astype(np.float32) / 255.0, blur=3)
    comp_roi = (rgb.astype(np.float32) * a[:, :, None] + roi.astype(np.float32) * (1 - a[:, :, None])).astype(np.uint8)
    comp = background_rgb.copy()
    comp[py1:py2, px1:px2] = comp_roi
    obj_mask = np.zeros((H, W), dtype=np.uint8)
    obj_mask[py1:py2, px1:px2] = (alpha > 15).astype(np.uint8) * 255
    # Boundary inpaint mask: ring around object, not the object core.
    kernel = np.ones((19, 19), np.uint8)
    dil = cv2.dilate(obj_mask, kernel, iterations=1)
    ero = cv2.erode(obj_mask, np.ones((7, 7), np.uint8), iterations=1)
    boundary = cv2.subtract(dil, ero)
    boundary = cv2.GaussianBlur(boundary, (0, 0), 2)
    pasted_box = [px1, py1, px2, py2]
    return comp, obj_mask, boundary, pasted_box

records = []
thumbs = []
for i, job in enumerate(jobs):
    jid = job['job_id']
    pid = job['prototype_id']
    layout_id = job['layout_id']
    lay = layout[layout_id]
    context_path = Path(lay['context_crop_5x'])
    bg = np.array(Image.open(context_path).convert('RGB'))
    rgba_path = CUTOUT_DIR / f'{pid}.png'
    rgba = np.array(Image.open(rgba_path).convert('RGBA'))
    target = lay['object_box_in_context_xyxy']
    comp, obj_mask, inpaint_mask, pasted_box = paste_rgba(bg, rgba, target, preserve_aspect=True)

    img_path = IMG_DIR / f'{jid}_{pid}_to_{layout_id}.jpg'
    obj_mask_path = MASK_DIR / f'{jid}_{pid}_to_{layout_id}.png'
    inpaint_mask_path = INPAINT_DIR / f'{jid}_{pid}_to_{layout_id}.png'
    Image.fromarray(comp).save(img_path, quality=95)
    Image.fromarray(obj_mask).save(obj_mask_path)
    Image.fromarray(inpaint_mask).save(inpaint_mask_path)

    overlay = Image.fromarray(comp).convert('RGB')
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([float(x) for x in target], outline=(255, 0, 0), width=3)
    draw.rectangle([float(x) for x in pasted_box], outline=(0, 255, 0), width=3)
    draw.text((8, 8), f'{jid} proto={pid} {job["prototype_grade"]} layout={layout_id}', fill=(255, 255, 0))
    overlay_path = OVERLAY_DIR / f'{jid}_{pid}_to_{layout_id}.jpg'
    overlay.save(overlay_path, quality=92)

    rec = {
        'job_id': jid,
        'prototype_id': pid,
        'prototype_grade': job['prototype_grade'],
        'layout_id': layout_id,
        'context_crop': str(context_path),
        'composite_image': str(img_path),
        'object_mask': str(obj_mask_path),
        'inpaint_boundary_mask': str(inpaint_mask_path),
        'overlay': str(overlay_path),
        'target_box_in_context_xyxy': target,
        'pasted_box_xyxy': pasted_box,
        'target_bbox_xywh': [target[0], target[1], target[2] - target[0], target[3] - target[1]],
        'pasted_bbox_xywh': [pasted_box[0], pasted_box[1], pasted_box[2] - pasted_box[0], pasted_box[3] - pasted_box[1]],
        'tag_overlap': job.get('tag_overlap', []),
    }
    records.append(rec)
    t = overlay.copy()
    t.thumbnail((360, 260))
    thumbs.append((jid, t))
    print(f'[{i+1:02d}/{len(jobs)}] {jid}: {pid} -> {layout_id}')

(OUT / 'composite_manifest.json').write_text(json.dumps({'count': len(records), 'records': records}, ensure_ascii=False, indent=2))

cols = 4
cell_w, cell_h = 400, 310
rows = int(np.ceil(len(thumbs) / cols))
sheet = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')
draw = ImageDraw.Draw(sheet)
for i, (jid, thumb) in enumerate(thumbs):
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h
    sheet.paste(thumb, (x + 20, y + 34))
    draw.text((x + 10, y + 8), jid, fill=(0, 0, 0))
sheet_path = OUT / 'composite_contact_sheet.jpg'
sheet.save(sheet_path, quality=92)

print(json.dumps({'out': str(OUT), 'count': len(records), 'contact_sheet': str(sheet_path), 'manifest': str(OUT / 'composite_manifest.json')}, ensure_ascii=False, indent=2))
