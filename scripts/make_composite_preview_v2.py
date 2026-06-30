
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

STAGE1 = Path('/root/workspace/outputs/arrester/stage1')
PROTO_BANK = Path('/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json')
MANIFEST = Path('/root/workspace/outputs/arrester/stage2/generation_manifest_v1.json')
OUT = Path('/root/workspace/outputs/arrester/stage2/composites_preview_v2')
IMG_DIR = OUT / 'images'
MASK_DIR = OUT / 'masks_object'
INPAINT_DIR = OUT / 'masks_inpaint_boundary'
ERASE_DIR = OUT / 'erased_backgrounds'
OVERLAY_DIR = OUT / 'overlays'
CUTOUT_DIR = Path('/root/workspace/outputs/arrester/prototypes/bank/cutouts_rgba')

for d in [IMG_DIR, MASK_DIR, INPAINT_DIR, ERASE_DIR, OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N = 40
manifest = json.loads(MANIFEST.read_text())
jobs = manifest['jobs'][:N]
layout = {x['layout_id']: x for x in json.loads((STAGE1 / 'layout_pool.json').read_text())['items']}


def tight_rgba(rgba, pad_ratio=0.06):
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return rgba
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    w, h = x2 - x1, y2 - y1
    pad_x = int(max(2, round(w * pad_ratio)))
    pad_y = int(max(2, round(h * pad_ratio)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(rgba.shape[1], x2 + pad_x)
    y2 = min(rgba.shape[0], y2 + pad_y)
    return rgba[y1:y2, x1:x2]


def feather(a, sigma=1.2):
    af = a.astype(np.float32) / 255.0
    af = cv2.GaussianBlur(af, (0, 0), sigma)
    return np.clip(af, 0, 1)


def erase_target(bg_rgb, target_box):
    h, w = bg_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in target_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), dtype=np.uint8)
    pad_x = max(8, int((x2 - x1) * 0.08))
    pad_y = max(8, int((y2 - y1) * 0.05))
    mask[max(0, y1-pad_y):min(h, y2+pad_y), max(0, x1-pad_x):min(w, x2+pad_x)] = 255
    # Telea only gives a rough clean plate. SD2 will later harmonize this region.
    bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), mask


def color_match(src_rgb, dst_rgb, alpha):
    mask = alpha > 20
    if mask.sum() < 20:
        return src_rgb
    src = src_rgb.astype(np.float32)
    dst = dst_rgb.astype(np.float32)
    out = src.copy()
    dmask = np.ones(dst.shape[:2], dtype=bool)
    for c in range(3):
        s = src[:, :, c][mask]
        d = dst[:, :, c][dmask]
        sm, ss = s.mean(), s.std() + 1e-6
        dm, ds = d.mean(), d.std() + 1e-6
        matched = (src[:, :, c] - sm) / ss * min(ds, ss * 1.2) + dm
        out[:, :, c] = src[:, :, c] * 0.6 + matched * 0.4
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_exact(bg_rgb, tight, target_box):
    H, W = bg_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in target_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    tw, th = max(1, x2 - x1), max(1, y2 - y1)
    resized = cv2.resize(tight, (tw, th), interpolation=cv2.INTER_AREA if tw < tight.shape[1] else cv2.INTER_CUBIC)
    rgb = resized[:, :, :3]
    alpha = resized[:, :, 3]
    roi = bg_rgb[y1:y2, x1:x2].copy()
    rgb = color_match(rgb, roi, alpha)
    a = feather(alpha, sigma=1.0)
    comp_roi = (rgb.astype(np.float32) * a[:, :, None] + roi.astype(np.float32) * (1 - a[:, :, None])).astype(np.uint8)
    comp = bg_rgb.copy()
    comp[y1:y2, x1:x2] = comp_roi
    obj_mask = np.zeros((H, W), dtype=np.uint8)
    obj_mask[y1:y2, x1:x2] = (alpha > 15).astype(np.uint8) * 255
    return comp, obj_mask, [x1, y1, x2, y2]

records=[]
thumbs=[]
for i, job in enumerate(jobs):
    jid=job['job_id']
    pid=job['prototype_id']
    layout_id=job['layout_id']
    lay=layout[layout_id]
    bg=np.array(Image.open(lay['context_crop_5x']).convert('RGB'))
    target=lay['object_box_in_context_xyxy']
    erased, erase_mask = erase_target(bg, target)
    rgba=np.array(Image.open(CUTOUT_DIR / f'{pid}.png').convert('RGBA'))
    tight=tight_rgba(rgba)
    comp, obj_mask, pasted_box = paste_exact(erased, tight, target)

    # Inpaint mask for SD2: erased target neighborhood + boundary ring, but protect object core.
    kernel_big=np.ones((25,25),np.uint8)
    kernel_small=np.ones((9,9),np.uint8)
    dil_obj=cv2.dilate(obj_mask,kernel_big,iterations=1)
    erode_obj=cv2.erode(obj_mask,kernel_small,iterations=1)
    boundary=cv2.subtract(dil_obj, erode_obj)
    inpaint_mask=cv2.bitwise_or(erase_mask, boundary)
    protected=cv2.erode(obj_mask, np.ones((15,15),np.uint8), iterations=1)
    inpaint_mask[protected>0]=0
    inpaint_mask=cv2.GaussianBlur(inpaint_mask,(0,0),2)

    stem=f'{jid}_{pid}_to_{layout_id}'
    img_path=IMG_DIR/f'{stem}.jpg'
    obj_path=MASK_DIR/f'{stem}.png'
    inp_path=INPAINT_DIR/f'{stem}.png'
    erased_path=ERASE_DIR/f'{stem}.jpg'
    Image.fromarray(comp).save(img_path,quality=95)
    Image.fromarray(obj_mask).save(obj_path)
    Image.fromarray(inpaint_mask).save(inp_path)
    Image.fromarray(erased).save(erased_path,quality=90)

    overlay=Image.fromarray(comp).convert('RGB')
    draw=ImageDraw.Draw(overlay)
    draw.rectangle([float(x) for x in target], outline=(255,0,0), width=3)
    draw.rectangle([float(x) for x in pasted_box], outline=(0,255,0), width=3)
    draw.text((8,8),f'{jid} proto={pid} {job["prototype_grade"]} layout={layout_id}',fill=(255,255,0))
    overlay_path=OVERLAY_DIR/f'{stem}.jpg'
    overlay.save(overlay_path,quality=92)

    records.append({
        'job_id':jid,
        'prototype_id':pid,
        'prototype_grade':job['prototype_grade'],
        'layout_id':layout_id,
        'context_crop':lay['context_crop_5x'],
        'erased_background':str(erased_path),
        'composite_image':str(img_path),
        'object_mask':str(obj_path),
        'inpaint_mask':str(inp_path),
        'overlay':str(overlay_path),
        'target_box_in_context_xyxy':target,
        'pasted_box_xyxy':pasted_box,
        'target_bbox_xywh':[target[0],target[1],target[2]-target[0],target[3]-target[1]],
        'pasted_bbox_xywh':[pasted_box[0],pasted_box[1],pasted_box[2]-pasted_box[0],pasted_box[3]-pasted_box[1]],
        'paste_policy':'tight_sam_cutout_resized_exactly_to_target_bbox_after_erasing_original_target'
    })
    t=overlay.copy(); t.thumbnail((360,260)); thumbs.append((jid,t))
    print(f'[{i+1:02d}/{len(jobs)}] {jid}: {pid} -> {layout_id}')

(OUT/'composite_manifest.json').write_text(json.dumps({'count':len(records),'records':records},ensure_ascii=False,indent=2))
cols=4; cell_w=400; cell_h=310
rows=int(np.ceil(len(thumbs)/cols))
sheet=Image.new('RGB',(cols*cell_w, rows*cell_h),'white')
draw=ImageDraw.Draw(sheet)
for i,(jid,thumb) in enumerate(thumbs):
    x=(i%cols)*cell_w; y=(i//cols)*cell_h
    sheet.paste(thumb,(x+20,y+34))
    draw.text((x+10,y+8),jid,fill=(0,0,0))
sheet_path=OUT/'composite_contact_sheet.jpg'
sheet.save(sheet_path,quality=92)
print(json.dumps({'out':str(OUT),'count':len(records),'contact_sheet':str(sheet_path),'manifest':str(OUT/'composite_manifest.json')},ensure_ascii=False,indent=2))
