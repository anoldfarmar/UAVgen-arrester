
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

STAGE1 = Path('/root/workspace/outputs/arrester/stage1')
MANIFEST = Path('/root/workspace/outputs/arrester/stage2/generation_manifest_v1.json')
CUTOUT_DIR = Path('/root/workspace/outputs/arrester/prototypes/bank/cutouts_rgba')
OUT = Path('/root/workspace/outputs/arrester/stage2/patch_inpaint_preview_v3')
PATCH_COMP_DIR = OUT / 'patch_composites'
PATCH_INP_DIR = OUT / 'patch_inpainted'
FULL_DIR = OUT / 'full_context_inpainted'
OVERLAY_DIR = OUT / 'overlays'
COMPARE_DIR = OUT / 'compare'
MASK_DIR = OUT / 'inpaint_masks'
for d in [PATCH_COMP_DIR, PATCH_INP_DIR, FULL_DIR, OVERLAY_DIR, COMPARE_DIR, MASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = '/root/autodl-tmp/diffusion'
N = 12
STRENGTH = 0.18
STEPS = 24
GUIDANCE = 5.5
PROMPT = 'aerial inspection photo of electrical power equipment, realistic local lighting, natural background transition, sharp details'
NEGATIVE = 'distorted arrester, broken object, duplicated object, blurry, unrealistic shadow, extra object, deformed structure, melted object'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

jobs = json.loads(MANIFEST.read_text())['jobs'][:N]
layout = {x['layout_id']: x for x in json.loads((STAGE1 / 'layout_pool.json').read_text())['items']}


def tight_rgba(rgba, pad_ratio=0.04):
    a = rgba[:, :, 3]
    ys, xs = np.where(a > 10)
    if len(xs) == 0:
        return rgba
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    w, h = x2 - x1, y2 - y1
    px = max(2, int(w * pad_ratio)); py = max(2, int(h * pad_ratio))
    return rgba[max(0,y1-py):min(rgba.shape[0],y2+py), max(0,x1-px):min(rgba.shape[1],x2+px)]


def crop_patch_box(target, W, H):
    x1,y1,x2,y2 = target
    tw, th = x2-x1, y2-y1
    cx, cy = (x1+x2)/2, (y1+y2)/2
    size = int(max(768, th*1.8, tw*4.0))
    size = min(size, 1152, max(W, H))
    pw = min(size, W); ph = min(size, H)
    # If target is extremely tall, allow rectangular patch but still capped.
    ph = min(max(ph, int(th + 260)), H, 1280)
    pw = min(max(pw, int(tw*4 + 260)), W, 1024)
    x1p = int(round(cx - pw/2)); y1p = int(round(cy - ph/2))
    x1p = max(0, min(W-pw, x1p)); y1p = max(0, min(H-ph, y1p))
    return [x1p, y1p, x1p+pw, y1p+ph]


def erase_target(patch, target_local):
    h,w = patch.shape[:2]
    x1,y1,x2,y2 = [int(round(v)) for v in target_local]
    mask = np.zeros((h,w), dtype=np.uint8)
    pad_x = max(8, int((x2-x1)*0.12)); pad_y = max(8, int((y2-y1)*0.06))
    mask[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)] = 255
    cleaned = cv2.inpaint(cv2.cvtColor(patch, cv2.COLOR_RGB2BGR), mask, 7, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), mask


def color_match(src, dst, alpha):
    m = alpha > 20
    if m.sum() < 20:
        return src
    s = src.astype(np.float32); d = dst.astype(np.float32); out = s.copy()
    # Match to narrow local ROI, conservatively.
    for c in range(3):
        sv = s[:,:,c][m]
        dv = d[:,:,c].reshape(-1)
        sm, ss = sv.mean(), sv.std()+1e-6
        dm, ds = dv.mean(), dv.std()+1e-6
        matched = (s[:,:,c]-sm)/ss*min(ds, ss*1.15)+dm
        out[:,:,c] = s[:,:,c]*0.72 + matched*0.28
    return np.clip(out,0,255).astype(np.uint8)


def paste_keep_aspect(bg, rgba_tight, target_local):
    H,W = bg.shape[:2]
    x1,y1,x2,y2 = [int(round(v)) for v in target_local]
    tw, th = max(1,x2-x1), max(1,y2-y1)
    sh, sw = rgba_tight.shape[:2]
    # Match object height to target bbox, preserve width. If width becomes wildly larger, cap gently.
    # Match target height while preserving aspect, but cap to patch dimensions.
    scale_h = th / max(1, sh)
    max_w = max(1, int(round(tw * 1.15)))
    scale_w_cap = max_w / max(1, sw)
    scale_patch_cap = min((W * 0.95) / max(1, sw), (H * 0.95) / max(1, sh))
    scale = min(scale_h, scale_w_cap, scale_patch_cap)
    nw, nh = max(1, int(round(sw*scale))), max(1, int(round(sh*scale)))
    px1 = int(round((x1+x2)/2 - nw/2)); py1 = int(round((y1+y2)/2 - nh/2))
    px1 = max(0, min(W-nw, px1)); py1 = max(0, min(H-nh, py1))
    px2, py2 = px1+nw, py1+nh
    resized = cv2.resize(rgba_tight, (nw,nh), interpolation=cv2.INTER_AREA if nw<sw else cv2.INTER_CUBIC)
    rgb = resized[:,:,:3]; alpha = resized[:,:,3]
    roi = bg[py1:py2, px1:px2].copy()
    rgb = color_match(rgb, roi, alpha)
    a = cv2.GaussianBlur(alpha.astype(np.float32)/255.0, (0,0), 0.8)
    comp = bg.copy()
    comp[py1:py2, px1:px2] = (rgb.astype(np.float32)*a[:,:,None] + roi.astype(np.float32)*(1-a[:,:,None])).astype(np.uint8)
    obj_mask = np.zeros((H,W), dtype=np.uint8)
    obj_mask[py1:py2, px1:px2] = (alpha>15).astype(np.uint8)*255
    return comp, obj_mask, [px1,py1,px2,py2]


def make_inpaint_mask(erase_mask, obj_mask):
    # Only target residual area and a thin ring around object, protect main body.
    dil = cv2.dilate(obj_mask, np.ones((17,17),np.uint8), iterations=1)
    ero = cv2.erode(obj_mask, np.ones((9,9),np.uint8), iterations=1)
    ring = cv2.subtract(dil, ero)
    mask = cv2.bitwise_or(erase_mask, ring)
    protected = cv2.dilate(ero, np.ones((7,7),np.uint8), iterations=1)
    mask[protected>0] = 0
    return cv2.GaussianBlur(mask, (0,0), 1.5)


def crop_mult8(im):
    w,h = im.size
    nw, nh = (w//8)*8, (h//8)*8
    return im.crop((0,0,nw,nh)), (nw,nh)

print('loading SD2')
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE, local_files_only=True,
    safety_checker=None, requires_safety_checker=False,
)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
try: pipe.set_progress_bar_config(disable=True)
except Exception: pass

records=[]; thumbs=[]
for i,job in enumerate(jobs):
    jid=job['job_id']; pid=job['prototype_id']; layout_id=job['layout_id']
    lay=layout[layout_id]
    context=np.array(Image.open(lay['context_crop_5x']).convert('RGB'))
    H,W=context.shape[:2]
    target=lay['object_box_in_context_xyxy']
    patch_box=crop_patch_box(target,W,H)
    x1p,y1p,x2p,y2p=patch_box
    patch=context[y1p:y2p, x1p:x2p].copy()
    target_local=[target[0]-x1p,target[1]-y1p,target[2]-x1p,target[3]-y1p]
    erased, erase_mask=erase_target(patch,target_local)
    rgba=np.array(Image.open(CUTOUT_DIR/f'{pid}.png').convert('RGBA'))
    tight=tight_rgba(rgba)
    comp_patch,obj_mask,pasted_box=paste_keep_aspect(erased,tight,target_local)
    inpaint_mask=make_inpaint_mask(erase_mask,obj_mask)

    comp_img=Image.fromarray(comp_patch)
    mask_img=Image.fromarray(inpaint_mask).convert('L')
    comp_fit, fit_size = crop_mult8(comp_img)
    mask_fit, _ = crop_mult8(mask_img)
    gen=torch.Generator(device=DEVICE).manual_seed(20260702+i) if DEVICE=='cuda' else torch.Generator().manual_seed(20260702+i)
    out=pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=comp_fit, mask_image=mask_fit,
             strength=STRENGTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen).images[0]

    # paste inpainted patch back into context
    out_np=np.array(out.convert('RGB'))
    full=context.copy()
    ph,pw=out_np.shape[:2]
    full[y1p:y1p+ph, x1p:x1p+pw]=out_np

    stem=f'{jid}_{pid}_to_{layout_id}'
    comp_path=PATCH_COMP_DIR/f'{stem}.jpg'; inp_path=PATCH_INP_DIR/f'{stem}.jpg'; full_path=FULL_DIR/f'{stem}.jpg'; mask_path=MASK_DIR/f'{stem}.png'
    Image.fromarray(comp_patch).save(comp_path,quality=95)
    out.save(inp_path,quality=95)
    Image.fromarray(full).save(full_path,quality=95)
    Image.fromarray(inpaint_mask).save(mask_path)

    overlay=Image.fromarray(full).convert('RGB'); draw=ImageDraw.Draw(overlay)
    draw.rectangle([float(x) for x in target], outline=(255,0,0), width=3)
    pasted_global=[pasted_box[0]+x1p,pasted_box[1]+y1p,pasted_box[2]+x1p,pasted_box[3]+y1p]
    draw.rectangle([float(x) for x in pasted_global], outline=(0,255,0), width=3)
    draw.text((8,8),f'{jid} v3 strength={STRENGTH}',fill=(255,255,0))
    overlay_path=OVERLAY_DIR/f'{stem}.jpg'; overlay.save(overlay_path,quality=92)

    # compare crop: composite patch | mask | inpaint patch | full overlay thumbnail
    c1=Image.fromarray(comp_patch); c2=Image.fromarray(inpaint_mask).convert('RGB'); c3=out.copy(); c4=overlay.copy()
    for c in [c1,c2,c3,c4]: c.thumbnail((300,240))
    canvas=Image.new('RGB',(1240,280),'white'); d=ImageDraw.Draw(canvas); d.text((10,6),f'{jid}: comp | mask | inpaint patch | full',fill=(0,0,0))
    x=10
    for c in [c1,c2,c3,c4]: canvas.paste(c,(x,30)); x+=310
    compare_path=COMPARE_DIR/f'{stem}.jpg'; canvas.save(compare_path,quality=92)

    records.append({**job,'patch_box_xyxy':patch_box,'target_local_xyxy':target_local,'pasted_box_local_xyxy':pasted_box,'pasted_box_global_xyxy':pasted_global,'patch_composite':str(comp_path),'patch_inpainted':str(inp_path),'full_context_inpainted':str(full_path),'overlay':str(overlay_path),'inpaint_mask':str(mask_path),'compare':str(compare_path),'strength':STRENGTH,'steps':STEPS,'guidance_scale':GUIDANCE,'paste_policy':'prototype_tight_cutout_height_matched_to_target_bbox_preserve_aspect_patch_only_inpaint'})
    t=overlay.copy(); t.thumbnail((360,260)); thumbs.append((jid,t))
    print(f'[{i+1:02d}/{len(jobs)}] {jid}: patch={patch_box} pasted_global={pasted_global}')

(OUT/'v3_manifest.json').write_text(json.dumps({'count':len(records),'records':records},ensure_ascii=False,indent=2))
cols=3; cell_w=400; cell_h=310; rows=(len(thumbs)+cols-1)//cols
sheet=Image.new('RGB',(cols*cell_w, rows*cell_h),'white'); draw=ImageDraw.Draw(sheet)
for i,(jid,t) in enumerate(thumbs):
    x=(i%cols)*cell_w; y=(i//cols)*cell_h; sheet.paste(t,(x+20,y+34)); draw.text((x+10,y+8),jid,fill=(0,0,0))
sheet_path=OUT/'v3_contact_sheet.jpg'; sheet.save(sheet_path,quality=92)
print(json.dumps({'out':str(OUT),'count':len(records),'contact_sheet':str(sheet_path),'manifest':str(OUT/'v3_manifest.json')},ensure_ascii=False,indent=2))
