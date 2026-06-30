# Experiment Report: Object-Preserved UAVGen Adaptation for Lightning Arresters

## 1. Goal

The goal is to adapt UAVGen-style data augmentation to a UAV power-inspection target: lightning arresters. The experiment does not attempt to fully reproduce UAVGen's FLUX/ControlNet training stack. Instead, it keeps the most important ideas:

- high-quality visual prototypes
- real layout/focal-region sampling
- detector feedback filtering
- diffusion-based visual harmonization

The main adaptation is **object preservation**: the arrester body is copied from real data and protected; diffusion is used only to repair boundaries, lighting, and background transitions.

## 2. Why Not Directly Generate Arresters with Inpainting?

Direct inpainting would require the model to hallucinate a complete arrester inside an empty bbox. This is risky for this task:

| Risk | Direct inpainting problem | Current mitigation |
|---|---|---|
| Thin structure | The arrester may become broken, melted, or bent | Use real prototype body |
| Professional category | SD2 has weak prior for power equipment details | Keep category appearance from real images |
| Confusion | Arrester may become insulator/tower hardware | Use manually reviewed A/B prototypes |
| Label reliability | Generated object may not match bbox | Layout controls bbox, paste controls object position |
| Weak teacher | Current detector is single-class and cannot filter insulator confusion | Reduce hallucination before teacher filtering |

Therefore SD2 inpainting is used as **harmonization**, not as object synthesis.

## 3. Stage1 Summary

Dataset:

```text
train: 85 images, 142 arrester objects
val:   21 images, 35 arrester objects
total: 106 images, 177 arrester objects
```

Teacher detector:

```text
/root/ultralytics/models/arrester.pt
```

Validation result:

```text
Precision: 0.988
Recall: 0.743
mAP50: 0.791
mAP50-95: 0.435
```

Interpretation:

- good enough as a weak teacher/filter
- not good enough as final automatic quality judge
- cannot detect insulator confusion because it is single-class

Stage1 outputs included:

```text
all_objects.jsonl
dataset_stats.json
layout_pool.json
focal_crops/
context_crops_5x/
teacher_feedback.jsonl
background_tags_auto.jsonl
```

## 4. Prototype Bank

Manual review file:

```text
prototype_candidate_review_strict21.xlsx
```

Final prototype bank:

```text
A-grade core prototypes: 13
B-grade backup/complex prototypes: 13
C-grade excluded: 3
Total usable: 26
```

Output:

```text
/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json
```

A-grade prototypes are used with higher sampling priority. B-grade prototypes are retained for complex/occluded scenarios at lower weight.

## 5. SAM Mask Generation

SAM checkpoint:

```text
/root/autodl-tmp/sam/sam_vit_h_4b8939.pth
```

Result:

```text
26 / 26 masks generated successfully
score min/mean/max: 0.939 / 0.971 / 0.985
quality flags: none
```

Sample outputs included in this repo:

```text
sample_results/mask_quality/mask_contact_sheet.jpg
sample_results/mask_quality/*.jpg
```

## 6. Composite/Inpainting Iterations and Problems

### v1 Composite Problem

Problem:

- The whole focal crop was resized into the target bbox.
- The actual arrester occupied only a small fraction of the crop.
- Result: pasted target was much too small.

Fix:

- Use SAM mask bounding box to extract a tight prototype cutout.

### v2 Composite Problem

Improvement:

- Tight SAM cutout was resized into the target bbox.
- Target region was erased before paste.

Problem:

- Exact bbox resizing distorted the prototype aspect ratio.
- Some objects looked stretched or compressed.

Fix:

- Preserve prototype aspect ratio.
- Match height first, cap width if necessary.

### v3 Inpainting Problem

Improvement:

- Only local patches were sent to SD2, not entire tall context images.
- Strength was reduced to 0.18.

Problem:

- The full generated patch was pasted back into context.
- This caused a visible rectangular patch artifact and apparent misalignment.

Fix:

- Blend back only pixels under the inpaint mask.
- Keep the rest of the original/composite context unchanged.

### v4 Current State

v4 fixes the full-rectangle paste artifact by:

```text
patch composite
+ inpaint mask
+ SD2 output
+ mask-only blending back into the full context
```

v4 is the current candidate version for visual inspection.

Key outputs:

```text
sample_results/inpaint_v4/v4_contact_sheet.jpg
sample_results/inpaint_v4/*.jpg
```

## 7. Current Assessment

What works:

- Prototype selection and mask extraction are stable.
- Layout/context sampling is ready.
- The pipeline is now structured into reproducible scripts.
- Major early bugs were identified and fixed: small paste, aspect distortion, whole-patch replacement.

What remains uncertain:

- v4 visual quality needs user confirmation.
- Some real layouts have occlusions or large target boxes that may still be hard to harmonize.
- SD2 inpainting may not be strong enough for all power-equipment contexts.
- Teacher detector can only be used for weak filtering.

## 8. Recommended Next Steps

1. Inspect v4 examples.
2. If acceptable, run v4 on the remaining 40 preview jobs or all 200 manifest jobs.
3. Run teacher detector filtering on generated samples.
4. Keep only high-confidence and visually plausible outputs.
5. Train detection model with a small synthetic ratio first.
6. Compare against real-only baseline.

## 9. Important Paths from Original Workspace

```text
/root/workspace/outputs/arrester/prototypes/bank/
/root/workspace/outputs/arrester/stage2/patch_inpaint_preview_v4/
/root/workspace/outputs/arrester/stage2/generation_manifest_v1.json
/root/workspace/plan/uavgen_paper_vs_arrester_experiment.md
/root/workspace/log/2026-06-29_arrester_uavgen_worklog.md
```

