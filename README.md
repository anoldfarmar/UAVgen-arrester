# UAVgen-arrester

Object-preserved diffusion augmentation experiment for UAV-based lightning arrester detection.

This repository contains the experimental scripts, configs, reports, and selected visual examples produced while adapting the CVPR 2026 UAVGen idea to a single professional class: **lightning arrester**.

## Motivation

UAVGen uses visual prototypes, focal regions, and detector feedback to improve UAV object detection data generation. For lightning arresters, directly asking a generic diffusion model to generate the object is risky because the object is thin, professional, and easily confused with insulators or tower hardware. This experiment therefore uses a more conservative pipeline:

```text
real arrester prototype
+ SAM mask
+ real layout/context crop
+ object-preserved paste
+ SD2 inpainting for boundary/background harmonization only
+ teacher detector filtering
```

## Repository Layout

```text
configs/              Pipeline path and generation configs
scripts/              Stage1/Stage2 data processing and generation scripts
requirements/         Diffusion environment requirements and install notes
docs/                 Experiment notes and paper comparison
sample_results/       Selected sample outputs for inspection
```

## External Assets Not Included

Large model/data files are intentionally not tracked.

Expected local paths used in the experiment:

```text
Teacher detector: /root/ultralytics/models/arrester.pt
SAM checkpoint:   /root/autodl-tmp/sam/sam_vit_h_4b8939.pth
SD2 inpainting:   /root/autodl-tmp/diffusion
Dataset config:   /root/autodl-tmp/bileiqi/data.yaml
```

## Environment

Detection was run in `yolo26`.

Diffusion/SAM was run in `diffusion`. See:

```text
requirements/requirements.txt
requirements/INSTALL.md
```

## Current Status

Completed:

- Stage1 dataset/statistics preparation
- Prototype bank construction from manual A/B/C review
- SAM mask generation for 26 prototypes
- Layout/context pool preparation from 177 annotations
- Composite preview v1/v2 experiments
- SD2 inpainting preview v1/v3/v4 experiments
- Problem analysis and method comparison document

Open issues:

- Composite/inpainting is still experimental and requires visual QC.
- v1/v2/v3 exposed scale, stretching, and rectangular patch artifact issues.
- v4 fixes full-patch replacement by blending only masked pixels back, but still needs user visual confirmation before generating larger batches.

## Key Reports

```text
docs/paper_vs_experiment.md
EXPERIMENT_REPORT.md
```

## Selected Visual Results

```text
sample_results/mask_quality/
sample_results/composite_v2/
sample_results/inpaint_v4/
```

