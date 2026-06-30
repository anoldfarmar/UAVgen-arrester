
import csv
import json
import shutil
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from collections import Counter

STAGE1 = Path('/root/workspace/outputs/arrester/stage1')
XLSX = STAGE1 / 'prototype_candidate_review_strict21.xlsx'
OUT = Path('/root/workspace/outputs/arrester/prototypes/bank')
IMG_OUT = OUT / 'images'
THUMB_OUT = OUT / 'thumbnails'
OUT.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)
THUMB_OUT.mkdir(parents=True, exist_ok=True)

NS = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def colrow(cell):
    col = ''.join(ch for ch in cell if ch.isalpha())
    row = int(''.join(ch for ch in cell if ch.isdigit()))
    n = 0
    for ch in col:
        n = n * 26 + ord(ch.upper()) - 64
    return row, n

def read_xlsx(path):
    with ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('a:si', NS):
                shared.append(''.join((t.text or '') for t in si.findall('.//a:t', NS)))
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = {}
        for c in sheet.findall('.//a:c', NS):
            ref = c.attrib.get('r')
            r, cidx = colrow(ref)
            val = ''
            v = c.find('a:v', NS)
            if v is not None:
                raw = v.text or ''
                val = shared[int(raw)] if c.attrib.get('t') == 's' and raw else raw
            elif c.attrib.get('t') == 'inlineStr':
                t = c.find('.//a:t', NS)
                val = t.text if t is not None else ''
            rows.setdefault(r, {})[cidx] = val
    maxcol = max(max(cols) for cols in rows.values())
    headers = [rows.get(1, {}).get(i, '') for i in range(1, maxcol + 1)]
    records = []
    for ridx in sorted(k for k in rows if k != 1):
        vals = [rows[ridx].get(i, '') for i in range(1, maxcol + 1)]
        if any(vals):
            records.append(dict(zip(headers, vals)))
    return records

def load_jsonl(path, key):
    data = {}
    if not path.exists():
        return data
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            data[r[key]] = r
    return data

# all object metadata, keyed by crop/layout id
objects = {}
with (STAGE1 / 'all_objects.jsonl').open() as f:
    for line in f:
        r = json.loads(line)
        crop_id = f"{r['split']}_{Path(r['image']).stem}_obj{r['object_index']:03d}"
        objects[crop_id] = r

background = load_jsonl(STAGE1 / 'background_tags_auto.jsonl', 'crop_id')
layout_items = {}
layout_path = STAGE1 / 'layout_pool.json'
if layout_path.exists():
    layout = json.loads(layout_path.read_text())
    for item in layout.get('items', []):
        layout_items[item['layout_id']] = item

raw = read_xlsx(XLSX)
seen = set()
prototypes = []
excluded = []
for row in raw:
    crop_id = (row.get('crop_id') or '').strip()
    grade = (row.get('manual_prototype_grade') or '').strip().upper()
    if not crop_id or crop_id in seen:
        continue
    seen.add(crop_id)
    obj = objects.get(crop_id)
    if obj is None:
        excluded.append({'crop_id': crop_id, 'reason': 'missing object metadata', 'row': row})
        continue
    source_crop = STAGE1 / 'focal_crops' / f'{crop_id}.jpg'
    if not source_crop.exists():
        excluded.append({'crop_id': crop_id, 'reason': 'missing focal crop', 'row': row})
        continue
    if grade not in {'A', 'B'}:
        excluded.append({'crop_id': crop_id, 'reason': f'excluded grade {grade or "blank"}', 'row': row})
        continue
    tier = 'core' if grade == 'A' else 'backup_complex'
    dst = IMG_OUT / f'{crop_id}.jpg'
    shutil.copy2(source_crop, dst)
    try:
        from PIL import Image, ImageDraw
        im = Image.open(source_crop).convert('RGB')
        im.thumbnail((320, 320))
        im.save(THUMB_OUT / f'{crop_id}.jpg', quality=90)
    except Exception:
        pass
    bg = background.get(crop_id, {})
    lay = layout_items.get(crop_id, {})
    proto = {
        'prototype_id': crop_id,
        'grade': grade,
        'tier': tier,
        'image': str(dst),
        'thumbnail': str(THUMB_OUT / f'{crop_id}.jpg'),
        'source_image': obj['image'],
        'object_index': obj['object_index'],
        'bbox_xyxy': obj['xyxy'],
        'bbox_xywh': obj['bbox_xywh'],
        'bbox_yolo': obj['yolo'],
        'area_ratio': obj['area_ratio'],
        'aspect_ratio': obj['aspect_ratio'],
        'background_tags_auto': bg.get('background_tags_auto', []),
        'background_scores': bg.get('background_scores', {}),
        'context_crop_5x': lay.get('context_crop_5x'),
        'context_crop_box_xyxy': lay.get('context_crop_box_xyxy'),
        'object_box_in_context_xyxy': lay.get('object_box_in_context_xyxy'),
        'teacher_iou': float(row['iou']) if str(row.get('iou','')).strip() else None,
        'teacher_confidence': float(row['confidence']) if str(row.get('confidence','')).strip() else None,
        'notes': row.get('notes', ''),
    }
    prototypes.append(proto)

bank = {
    'source_review_file': str(XLSX),
    'policy': {
        'A': 'core prototype, first-round generation priority',
        'B': 'backup/complex prototype, use at lower sampling weight',
        'C_or_blank': 'excluded from first prototype bank',
        'background_tags': 'auto_tags are weak hints only, not hard semantic labels',
    },
    'counts': dict(Counter(p['grade'] for p in prototypes)),
    'total': len(prototypes),
    'prototypes': prototypes,
    'excluded': excluded,
}
(OUT / 'prototype_bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2))
with (OUT / 'prototype_review_merged.csv').open('w', newline='') as f:
    fields = ['prototype_id','grade','tier','image','source_image','object_index','teacher_confidence','teacher_iou','area_ratio','aspect_ratio','background_tags_auto','context_crop_5x','notes']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for p in prototypes:
        row = {k: p.get(k) for k in fields}
        row['background_tags_auto'] = '|'.join(p.get('background_tags_auto') or [])
        w.writerow(row)
summary = {
    'out': str(OUT),
    'total': len(prototypes),
    'counts': bank['counts'],
    'excluded_count': len(excluded),
    'images_dir': str(IMG_OUT),
    'prototype_bank': str(OUT / 'prototype_bank.json'),
}
(OUT / 'README.md').write_text('# Arrester Prototype Bank\n\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
