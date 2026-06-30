# 避雷器 Object-Preserved UAVGen 实验报告

## 1. 实验目标

本实验目标是将 UAVGen 论文中的核心思想迁移到电力巡检场景中的 **避雷器检测数据增强**。

UAVGen 的核心思想包括：

- 使用高质量 visual prototype 作为生成条件；
- 使用真实 layout / focal region 控制目标位置和尺度；
- 使用 detector feedback 做生成后标签修正和质量过滤；
- 通过扩散模型提升生成图像自然度。

但避雷器属于专业细长目标，直接让通用扩散模型生成目标主体风险较高。因此本实验没有直接复现完整 FLUX/ControlNet 训练流程，而是采用更保守的工程路线：

```text
真实避雷器 prototype
+ SAM 精细 mask
+ 真实 layout/context crop
+ object-preserved paste
+ SD2 inpainting 只做边界和背景过渡修复
+ teacher detector 初筛
```

也就是说，本实验的核心原则是：

```text
避雷器主体来自真实图像，扩散模型只负责 harmonization，不负责重新创造避雷器。
```

## 2. 为什么不直接用 SD2 Inpainting 生成避雷器？

直接 inpainting 的做法通常是：在背景图中给一个空 mask 或 bbox，让 SD2 根据 prompt 生成一个避雷器。这个方法对通用物体可能可行，但对避雷器风险很高。

| 风险点 | 直接 inpainting 的问题 | 本实验的处理方式 |
|---|---|---|
| 目标结构细长 | 避雷器可能被生成断裂、弯曲、模糊或粘连 | 使用真实 prototype 保持主体结构 |
| 类别专业 | SD2 对电力设备细节先验不足 | 从真实避雷器中裁剪 prototype |
| 类别混淆 | 可能生成成绝缘子、金具或塔材的一部分 | 使用人工筛选的 A/B 级 prototype |
| 标签不可靠 | 生成目标可能不匹配预设 bbox | 贴入时由 layout 控制 bbox 和位置 |
| teacher 较弱 | 当前 detector 是单类避雷器模型，无法强判断绝缘子混淆 | 前端减少幻觉，后端只做弱过滤 |

因此，本实验使用 SD2 inpainting 的定位是：

```text
粗合成图 composite
↓
保护避雷器主体
↓
只在边界/周围背景区域 inpaint
↓
修复贴图边缘、光照、阴影和颜色过渡
```

这是一种 **diffusion harmonization**，不是纯文本引导的目标生成。

## 3. Stage 1：数据准备与 Teacher Detector 评估

### 3.1 数据规模

数据配置：

```text
/root/autodl-tmp/bileiqi/data.yaml
```

数据规模：

```text
train: 85 张图，142 个避雷器目标
val:   21 张图，35 个避雷器目标
总计:  106 张图，177 个避雷器目标
```

已完成：

- 标签合法性检查；
- bbox 宽高、面积、长宽比统计；
- 目标中心点热力图；
- focal crop 生成；
- context crop 5x 生成；
- layout pool 生成。

关键输出：

```text
/root/workspace/outputs/arrester/stage1/dataset_stats.json
/root/workspace/outputs/arrester/stage1/layout_pool.json
/root/workspace/outputs/arrester/stage1/focal_crops/
/root/workspace/outputs/arrester/stage1/context_crops_5x/
```

### 3.2 Teacher Detector

使用模型：

```text
/root/ultralytics/models/arrester.pt
```

验证集结果：

```text
Precision: 0.988
Recall: 0.743
mAP50: 0.791
mAP50-95: 0.435
```

评估结论：

- precision 较高，说明检测出来的避雷器大多可信；
- recall 一般，会漏掉部分真实避雷器；
- mAP50-95 不高，说明精细定位能力一般；
- 当前模型是单类模型，不能判断“避雷器 vs 绝缘子”的混淆。

因此该模型适合作为：

```text
弱 teacher / 初筛器
```

不适合作为：

```text
最终自动质量裁判
```

## 4. Prototype Bank 构建

### 4.1 初筛

根据 teacher feedback 生成严格候选：

```text
IoU >= 0.9
confidence >= 0.85
```

严格候选文件：

```text
prototype_candidates_strict.txt
prototype_candidate_review_strict21.csv
```

### 4.2 人工筛选

用户在 Windows 上标注：

```text
prototype_candidate_review_strict21.xlsx
```

最终标注结果：

```text
A: 13
B: 13
C: 3
```

含义：

| 等级 | 含义 | 使用策略 |
|---|---|---|
| A | 高质量核心 prototype | 第一轮优先使用 |
| B | 可用但存在遮挡/复杂背景 | 低权重使用 |
| C | 不适合作为 prototype | 第一轮排除 |

最终 prototype bank：

```text
/root/workspace/outputs/arrester/prototypes/bank/prototype_bank.json
```

仓库中保留了部分示例：

```text
sample_results/manifests/prototype_review_merged.csv
```

## 5. SAM Mask 生成

使用 SAM：

```text
/root/autodl-tmp/sam/sam_vit_h_4b8939.pth
```

结果：

```text
26 / 26 个 prototype 成功生成 mask
SAM score min/mean/max: 0.939 / 0.971 / 0.985
异常 flag: 0
```

输出：

```text
/root/workspace/outputs/arrester/prototypes/bank/masks/
/root/workspace/outputs/arrester/prototypes/bank/mask_overlays/
/root/workspace/outputs/arrester/prototypes/bank/cutouts_rgba/
```

仓库示例：

```text
sample_results/mask_quality/mask_contact_sheet.jpg
sample_results/mask_quality/*.jpg
```

结论：SAM 对当前 prototype 的分割质量整体可用，适合作为后续 object-preserved paste 的主体 mask。

## 6. Composite 与 Inpainting 迭代问题分析

本阶段进行了多轮实验，逐步暴露并修复问题。

### 6.1 Composite v1：目标过小

v1 做法：

```text
将整张 prototype focal crop 按比例缩放进目标 bbox
```

问题：

- prototype focal crop 中包含大量背景；
- 真实避雷器主体只占 crop 的一小部分；
- 缩放后贴入的避雷器明显小于真实 layout bbox。

现象：

```text
红框是目标 bbox，绿框是贴入范围，但真正避雷器只占很小区域。
```

修复：

```text
使用 SAM mask 的 tight cutout，而不是整张 focal crop。
```

### 6.2 Composite v2：尺度正确但存在拉伸风险

v2 做法：

```text
先擦除目标红框区域
再将 SAM tight cutout 精确缩放到目标 bbox
```

改进：

- 贴入目标大小和真实 bbox 一致；
- 避免了 v1 目标过小的问题。

问题：

- 如果 prototype 的长宽比和目标 bbox 不一致，强行填满 bbox 会导致横向或纵向拉伸；
- 部分避雷器看起来比例不自然。

修复：

```text
按目标高度优先缩放，保持 prototype 原始长宽比，不强制填满 bbox 宽度。
```

### 6.3 Inpainting v1：整张 context crop 送入 SD2 导致变糊

v1 inpainting 做法：

```text
将完整 context crop 直接送入 SD2 inpainting
```

问题：

- 部分 context crop 尺寸很高，例如 512x1848、704x2936；
- SD2 更适合局部固定尺度修复；
- 大尺寸输入导致局部模糊、比例怪、背景被重采样。

修复：

```text
只裁目标附近 patch 送入 SD2，不处理整张高图。
```

### 6.4 Patch Inpainting v3：矩形回贴错位

v3 做法：

```text
只裁目标附近 patch 做 SD2 inpainting
再将 SD2 输出 patch 回贴到 full context
```

改进：

- 不再直接处理完整大图；
- prototype 保持长宽比；
- denoise strength 降到 0.18。

问题：

- 回贴时错误地把 SD2 输出的整个 patch 矩形覆盖回 full context；
- 造成明显矩形块和错位感。

修复：

```text
只将 inpaint mask 区域的 SD2 像素融合回 full context，其余区域保持原图/粗合成不变。
```

### 6.5 Patch Inpainting v4：当前候选方案

v4 做法：

```text
局部 patch composite
+ inpaint mask
+ SD2 输出
+ 仅 mask 区域融合回 full context
```

修复的问题：

- 避免目标过小；
- 避免强行拉伸；
- 避免整张高图输入 SD2；
- 避免完整矩形 patch 回贴。

当前状态：

- v4 是当前最合理的候选版本；
- 仍需人工检查图像质量；
- 若质量可接受，可扩展到更多生成任务。

仓库示例：

```text
sample_results/inpaint_v4/v4_contact_sheet.jpg
sample_results/inpaint_v4/*.jpg
```

## 7. 当前实验结论

目前已经验证：

1. **prototype bank 构建是可行的**  
   通过 teacher + 人工筛选，可以获得一批相对可靠的避雷器主体。

2. **SAM mask 质量整体可用**  
   26 个 prototype 均成功生成 mask，数值质量稳定。

3. **直接粗贴图存在明显问题**  
   如果不使用 tight mask，会导致目标过小；如果强行填满 bbox，会导致比例变形。

4. **SD2 inpainting 只能作为局部修边工具**  
   不能指望它重新生成高质量避雷器主体，也不适合处理整张大尺寸 context crop。

5. **v4 是目前最合理的工程版本**  
   它保留真实主体，只在 mask 区域做局部 harmonization。

## 8. 当前不足

| 问题 | 当前状态 | 后续建议 |
|---|---|---|
| teacher detector 较弱 | recall 一般，且单类 | 只做弱过滤，不做最终裁判 |
| prototype 数量有限 | A=13，B=13 | 后续扩充更多高质量 prototype |
| 背景标签为 auto_tags | 未人工精修 | 第一版只做弱匹配，不做硬过滤 |
| inpainting 质量未最终确认 | v4 待人工检查 | 先小批量质检，再扩大生成 |
| 遮挡关系未建模 | 当前为简单贴入+修边 | 后续可加入遮挡/深度规则 |

## 9. 下一步建议

建议按如下顺序继续：

1. 人工检查 v4 输出结果；
2. 如果可接受，扩展到 40 张或 200 张 manifest；
3. 用 `/root/ultralytics/models/arrester.pt` 做 teacher 回检；
4. 保留：

```text
confidence >= 0.5
IoU >= 0.4
```

5. 对低置信/低 IoU 样本人工抽查；
6. 将保留样本加入检测训练；
7. 与真实数据 baseline 做 mAP/recall 对比。

## 10. 与 UAVGen 论文的关系

本实验不是完整复现 UAVGen 的 FLUX + ControlNet 训练流程，而是对其核心思想进行避雷器场景适配。

| UAVGen 思想 | 本实验对应实现 |
|---|---|
| visual prototype | A/B 级避雷器 prototype bank |
| focal region | focal crops + context_crops_5x |
| layout condition | layout_pool.json 中的真实 bbox/location |
| foreground quality | SAM mask + object-preserved paste |
| label refinement | teacher detector 回检与 IoU/confidence 筛选 |
| 生成质量控制 | 多阶段人工质检 + 小批量试验 |

关键差异：

```text
论文：prototype-conditioned diffusion 生成目标
本实验：真实 prototype 保持目标主体，SD2 只做局部修边
```

这个差异是有意设计的，目的是降低避雷器这种专业细长目标的类别幻觉、结构变形和标签不一致风险。

## 11. 重要路径

原始工作区路径：

```text
/root/workspace/outputs/arrester/prototypes/bank/
/root/workspace/outputs/arrester/stage2/patch_inpaint_preview_v4/
/root/workspace/outputs/arrester/stage2/generation_manifest_v1.json
/root/workspace/plan/uavgen_paper_vs_arrester_experiment.md
/root/workspace/log/2026-06-29_arrester_uavgen_worklog.md
```

仓库示例路径：

```text
sample_results/mask_quality/
sample_results/composite_v2/
sample_results/inpaint_v4/
docs/paper_vs_experiment.md
EXPERIMENT_REPORT.md
```
