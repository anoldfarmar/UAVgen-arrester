
import json
import random
from pathlib import Path
from collections import defaultdict

STAGE1 = Path('/root/workspace/outputs/arrester/stage1')
PROTO_BANK = Path('/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json')
OUT = Path('/root/workspace/outputs/arrester/stage2')
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 200
SEED = 20260629
random.seed(SEED)

bank = json.loads(PROTO_BANK.read_text())
prototypes = bank['prototypes']
layout = json.loads((STAGE1 / 'layout_pool.json').read_text())['items']
background = {}
with (STAGE1 / 'background_tags_auto.jsonl').open() as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        background[r['crop_id']] = r

# A prototypes are sampled more often; B is retained for complex/occluded contexts.
weighted_protos = []
for p in prototypes:
    weight = 5 if p['grade'] == 'A' else 2
    weighted_protos.extend([p] * weight)

def tag_set(x):
    return set(x or [])

def choose_background(proto_tags):
    candidates = []
    pset = tag_set(proto_tags)
    for item in layout:
        cid = item['layout_id']
        bg = background.get(cid, {})
        btags = tag_set(bg.get('background_tags_auto', []))
        overlap = len(pset & btags)
        # weak matching only. Unknown/low-overlap contexts remain valid to preserve diversity.
        score = overlap + (0.25 if 'sky' in btags else 0) + random.random() * 0.05
        candidates.append((score, item, bg))
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:max(20, len(candidates)//3)]
    return random.choice(top)[1:]

jobs = []
for idx in range(N_JOBS):
    proto = random.choice(weighted_protos)
    layout_item, bg = choose_background(proto.get('background_tags_auto', []))
    # Use real layout bbox from the background/context image. The prototype will be scaled into this bbox.
    job = {
        'job_id': f'gen_{idx:04d}',
        'seed': SEED + idx,
        'prototype_id': proto['prototype_id'],
        'prototype_grade': proto['grade'],
        'prototype_image': proto['image'],
        'prototype_source_image': proto['source_image'],
        'layout_id': layout_item['layout_id'],
        'target_image': layout_item['image'],
        'target_context_crop_5x': layout_item.get('context_crop_5x'),
        'target_bbox_xyxy': layout_item['bbox_xyxy'],
        'target_bbox_xywh': layout_item['bbox_xywh'],
        'target_bbox_yolo': layout_item['bbox_yolo'],
        'target_image_width': layout_item['image_width'],
        'target_image_height': layout_item['image_height'],
        'background_tags_auto': bg.get('background_tags_auto', []),
        'prototype_tags_auto': proto.get('background_tags_auto', []),
        'tag_overlap': sorted(list(tag_set(bg.get('background_tags_auto', [])) & tag_set(proto.get('background_tags_auto', [])))),
        'prompt': 'an aerial inspection image of electrical power equipment, realistic lighting, natural background transition',
        'negative_prompt': 'distorted arrester, broken object, duplicated object, blurry, unrealistic shadow, extra object',
        'harmonization': {
            'mode': 'object_preserved_inpainting',
            'denoise_strength_initial': 0.25,
            'denoise_strength_range': [0.15, 0.35],
            'mask_policy': 'protect prototype core; inpaint feathered boundary and nearby background only'
        }
    }
    jobs.append(job)

manifest = {
    'description': 'First stage2 generation manifest. Uses prototype bank + all real layout/context annotations. No images are generated here.',
    'seed': SEED,
    'num_jobs': len(jobs),
    'prototype_bank': str(PROTO_BANK),
    'layout_pool': str(STAGE1 / 'layout_pool.json'),
    'jobs': jobs,
}
(OUT / 'generation_manifest_v1.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
# smaller review CSV
with (OUT / 'generation_manifest_v1_preview.csv').open('w') as f:
    f.write('job_id,prototype_id,prototype_grade,layout_id,target_context_crop_5x,tag_overlap\n')
    for j in jobs:
        f.write(','.join([j['job_id'], j['prototype_id'], j['prototype_grade'], j['layout_id'], j['target_context_crop_5x'] or '', '|'.join(j['tag_overlap'])]) + '\n')
print(json.dumps({'manifest': str(OUT / 'generation_manifest_v1.json'), 'preview': str(OUT / 'generation_manifest_v1_preview.csv'), 'jobs': len(jobs)}, ensure_ascii=False, indent=2))
