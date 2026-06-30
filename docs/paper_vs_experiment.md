# UAVGen 论文方法与避雷器增强实验对照说明

## 参考来源

- 本地论文：`/root/workspace/paper/Li_Visual_Prototype_Conditioned_Focal_Region_Generation_for_UAV-Based_Object_Detection_CVPR_2026_paper.pdf`
- arXiv 页面：https://arxiv.org/abs/2604.02966
- 官方仓库：https://github.com/Sirius-Li/UAVGen
- 当前实验计划：`/root/workspace/plan/uavgen_crane_plan.md`
- 当前实验日志：`/root/workspace/log/2026-06-29_arrester_uavgen_worklog.md`

## 一句话结论

本实验和 UAVGen 论文的共同点是：都以 **高质量 visual prototype + 真实 layout/focal region + detector feedback label refinement** 为核心，目标都是生成对检测训练有用、标签可靠的 UAV 数据。

关键不同点是：论文训练一个 prototype-conditioned layout-to-image diffusion model，让扩散模型根据 prototype layout 生成目标；本实验针对避雷器这种细长、专业、易与绝缘子/塔材混淆的目标，采用更保守的 **object-preserved composite + SD2 inpainting harmonization**：真实避雷器主体不交给扩散模型重画，SD2 只负责边界、光照和背景过渡修复。

## 方法模块对照表

| 模块 | UAVGen 论文做法 | 本避雷器实验做法 | 共同点 | 主要差异与原因 |
|---|---|---|---|---|
| 任务目标 | 面向 UAV 通用目标检测数据增强，在 VisDrone/UAVDT 等公开数据集上提升检测 AP | 面向电力巡检 UAV 图像中的避雷器类别增强 | 都是为目标检测生成带标签训练数据 | 本实验是单一专业类别，类别形态更细、更窄，生成错误风险更高 |
| 数据生成范式 | layout-to-image generation：输入真实布局和多源条件，由扩散模型生成图像 | object-preserved generation：真实避雷器 prototype 先贴入真实 layout/context，再用 SD2 inpainting 修边 | 都利用真实 layout 控制目标位置和标签 | 论文让扩散模型生成目标；本实验保留真实目标主体，减少类别/结构幻觉 |
| Visual Prototype | 通过 detector 置信度、IoU、latent centroid 等规则选高质量 visual prototype | 用 `/root/ultralytics/models/arrester.pt` 初筛，再人工标 A/B/C；当前 A=13、B=13、C=3 | 都重视高质量 prototype，避免低质量 crop 干扰生成 | 本实验加入人工专业判断，因为电塔遮挡、避雷器与绝缘子/塔材粘连问题 detector 难以完全判断 |
| Prototype 使用方式 | 将 prototype 放到 blank canvas 的 layout 位置，编码成视觉条件，注入 diffusion/ControlNet | 将 SAM mask 后的真实 prototype 作为可贴入主体，直接形成 composite 图 | 都将 prototype 与目标 layout 对齐 | 论文使用 prototype 作为条件信号；本实验把 prototype 本体作为最终目标主体 |
| Diffusion 模型 | 基于 FLUX，训练 VPC-DM；视觉 prototype layout embedding + text embedding + ControlNet 注入 | 使用本地 `stable-diffusion-2-inpainting`，不训练完整 VPC-DM，做局部 harmonization | 都使用扩散模型增强图像自然性 | 本实验数据量小、类别专业，不适合从零/重训练完整生成模型；SD2 只做局部修复 |
| 文本条件 | 全局 prompt 和对象级 prompt，例如 aerial image with classes / aerial image of class | prompt 偏向电力巡检场景和自然过渡，例如 aerial inspection image of electrical power equipment | 都有文本语义引导 | 本实验 prompt 不强调“重新生成避雷器”，避免模型改写主体 |
| Focal Region | FRE-DP：根据目标中心聚类，裁剪目标密集区域，生成后再合回原图 | 已生成 focal crops 和 `context_crops_5x`，使用真实 layout_pool 的 177 个标注采样 | 都避免把模型容量浪费在大面积无目标背景上 | 本实验先做局部 composite/inpainting，小批量质检后再扩展 |
| 前景关注 | 论文使用 foreground-aware reweighted loss，让目标区域损失权重更高 | 本实验用 mask policy 保护目标主体，并让 inpainting mask 主要覆盖边界区域 | 都强调目标区域质量 | 论文通过训练损失实现；本实验通过 mask 保护和低 denoise strength 实现 |
| Label Refinement | 生成后用 pretrained detector 做 IoU matching，处理 missed generation、false generation、label misalignment | 生成后计划用 `arrester.pt` 做 confidence/IoU 回检，低质量丢弃或复查 | 都依赖 detector feedback 修正/过滤生成数据 | 本实验 teacher 是单类弱 teacher，不能做绝缘子混淆强判断，所以更依赖 object-preserved 方案和人工抽检 |
| 生成风险 | 小目标模糊、layout 边界 artifact、漏生成、多生成、标签错位 | 避雷器主体变形、被画成绝缘子/塔材、边界贴图痕迹、遮挡关系不自然 | 都关注图像-标签一致性 | 本实验最大的风险是专业目标结构被扩散模型改写，因此不让 diffusion 负责主体生成 |
| 当前实验阶段 | 论文完整训练 VPC-DM + FRE-DP + label refinement | 已完成 stage1、prototype bank、layout pool、SAM mask；下一步做少量 composite 和 SD2 inpainting 质检 | 都按 prototype/focal/refinement 思路推进 | 本实验是工程复现/适配，不是完整论文训练复现 |

## 为什么避雷器不直接用 inpainting 生成目标？

直接 inpainting 生成避雷器的流程通常是：在背景图上给一个空 mask 或 bbox，让 SD2 根据 prompt 在该区域生成一个“避雷器”。这对通用物体可能可行，但对本实验风险很高。

| 风险点 | 直接 inpainting 生成避雷器的问题 | 本实验采用 object-preserved 的原因 |
|---|---|---|
| 类别专业性 | SD2 是通用 inpainting 模型，对电力巡检中的避雷器形态先验不足 | 使用真实避雷器 prototype，类别主体天然正确 |
| 结构细长 | 避雷器通常细长、局部纹理弱，扩散模型容易生成断裂、弯曲、粘连结构 | SAM mask 抠出真实主体，结构不交给扩散模型重画 |
| 易混淆 | 避雷器可能被生成成绝缘子、金具、塔材的一部分 | 真实 prototype 减少“生成成别的设备”的概率 |
| 标签可靠性 | 纯 inpainting 后生成物的位置、大小、形状可能与预设 bbox 不一致 | 贴入时 bbox 可控，标签来自已知 layout 和 prototype 变换 |
| 小数据问题 | 当前只有少量避雷器数据和 26 个可用 prototype，不足以训练强 domain diffusion | 不训练完整生成模型，先利用通用 SD2 做局部视觉融合 |
| 后续过滤压力 | 如果主体由模型生成，teacher detector 需要承担强质量裁判；当前 `arrester.pt` recall 和定位能力有限 | 先保证主体真实，再让 teacher 只做弱过滤和对齐检查 |

因此，本实验中的 SD2 inpainting 不是用来“创造避雷器主体”，而是用来做：

```text
粗贴图 composite
↓
保护避雷器主体 mask
↓
只在边界/周围背景区域 inpaint
↓
修复贴图边缘、光照、阴影、颜色过渡
```

这是一种 **diffusion harmonization**，不是纯粹的 **text-guided object generation**。

## 论文如何处理类似问题？

论文并没有简单地让扩散模型凭空根据 bbox 生成目标。它也意识到 UAV 场景中小目标、遮挡和重叠会导致低质量布局条件、模糊目标和标签不一致。论文的处理方式主要有三层：

| 论文中的问题 | UAVGen 的处理方式 | 对本实验的启发 |
|---|---|---|
| 小目标/遮挡导致 crop 噪声大、视觉布局质量差 | 用 detector 的置信度和 IoU 先筛选清晰、定位准确的候选 prototype，再用 latent centroid 筛掉偏离类别中心的样本 | 我们用 teacher feedback + 人工 A/B/C 筛选 26 个 prototype，避免遮挡严重样本进入第一批 bank |
| bbox layout 信息过抽象，难以给出细粒度外观 | 将 visual prototype 放到 blank canvas 的布局位置，再编码成 visual-prototype-enhanced layout embedding | 我们同样使用 prototype + layout，但进一步把真实 prototype 保留为主体，而不只是作为条件 |
| 扩散生成容易在目标边界产生 artifact | 引入 high-quality visual prototype、多源条件、foreground-aware reweighted loss，提高目标区域生成质量 | 我们用 SD2 inpainting 只修边界，直接针对 copy-paste 边界 artifact 做 harmonization |
| 大图中目标稀疏，扩散容量浪费在背景 | FRE-DP 裁剪目标密集 focal region，训练/生成集中在高信息区域 | 我们生成 focal crops、context_crops_5x 和 layout_pool，先在局部区域质检和生成 |
| 生成图和标签不一致：漏生成、多生成、错位 | Label refinement：用 pretrained detector 做 IoU matching，丢弃 missed，加入 confident false detections，并在高置信时用 detector bbox 修正 misalignment | 我们计划用 `arrester.pt` 做 confidence/IoU 回检；但因 teacher 是弱单类模型，所以前面更保守地保持真实主体 |

也就是说，论文对“直接生成不可靠”的处理不是放弃扩散，而是通过 **高质量 prototype 条件 + focal region + detector label refinement** 降低风险。本实验继承这个思想，但由于避雷器比 VisDrone 通用类别更专业、更细、更容易被结构性误画，所以进一步把策略改为 **object-preserved**：目标主体真实，扩散只做局部修复。

## 本实验当前已完成内容与论文模块映射

| 当前产物 | 路径 | 对应论文模块 | 状态 |
|---|---|---|---|
| 数据统计与 bbox/layout 分布 | `/root/workspace/outputs/arrester/stage1/dataset_stats.json` | layout distribution / focal region 基础 | 已完成 |
| focal crops | `/root/workspace/outputs/arrester/stage1/focal_crops/` | focal region / prototype crop | 已完成 |
| context crops 5x | `/root/workspace/outputs/arrester/stage1/context_crops_5x/` | focal region 上下文 | 已完成 |
| layout pool | `/root/workspace/outputs/arrester/stage1/layout_pool.json` | 真实 layout 采样 | 已完成，177 个目标 |
| teacher feedback | `/root/workspace/outputs/arrester/stage1/teacher_feedback.jsonl` | detector-based prototype selection / label refinement | 已完成 |
| prototype bank | `/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json` | visual prototype set | 已完成，A=13、B=13 |
| SAM masks | `/root/workspace/outputs/arrester/prototypes/bank/masks/` | 本实验新增，用于 object-preserved cutout | 已完成，26/26 成功 |
| mask 质检拼图 | `/root/workspace/outputs/arrester/prototypes/bank/mask_contact_sheet.jpg` | 质量控制 | 待人工确认 |
| generation manifest | `/root/workspace/outputs/arrester/stage2/generation_manifest_v1.json` | layout/prototype 组合计划 | 已完成，200 个 job |

## 接下来建议的质检流程

为避免低质量数据进入扩散阶段，建议按以下检查点推进：

| 检查点 | 需要看的文件 | 通过标准 | 不通过时处理 |
|---|---|---|---|
| SAM mask 质量 | `mask_contact_sheet.jpg`、`mask_overlays/` | mask 覆盖避雷器主体，不过多包含塔材/背景 | 删除或降级对应 prototype；必要时人工修 mask |
| Composite 粗贴图 | `stage2/composites_preview/` | 位置、尺度、方向基本合理；没有明显离谱贴图 | 调整 layout 采样、缩放策略或 prototype 权重 |
| SD2 inpainting 修边 | `stage2/inpainted_preview/` | 边界更自然，主体未被改坏 | 降低 denoise strength，缩小 inpaint mask |
| Teacher 回检 | `stage2/filter_report.json` | confidence 和 IoU 达标 | 丢弃或进入人工复查 |
| 检测训练验证 | 后续训练报告 | 真实 val 上 recall/AP 提升，precision 不明显下降 | 降低合成比例，提高过滤阈值 |

## 最终定位

本实验不是严格复刻 UAVGen 的完整 FLUX + ControlNet 训练，而是面向避雷器类别做工程化适配。它保留了论文最关键的三件事：

1. **高质量 visual prototype**：用 teacher detector + 人工筛选构建 prototype bank。
2. **真实 layout / focal region**：用 177 个真实标注和局部 context 做生成约束。
3. **detector feedback refinement**：生成后用 teacher detector 做回检和过滤。

同时，本实验对避雷器做了更保守的改造：

```text
论文：prototype-conditioned diffusion 生成目标
本实验：真实 prototype 保持目标主体，SD2 inpainting 只做局部 harmonization
```

这个差异是有意设计的，不是偏离目标。它是为了降低专业细长目标的类别幻觉、结构变形和标签不一致风险。
