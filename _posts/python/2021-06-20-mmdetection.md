---
layout: post
title: 'MMDetection 用户笔记'
date: 2021-06-20
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/6554370ac458853aef3d6601.jpg'
tags: Python
---

> Notes about MMDetection.

# 1. 训练模型

1. `tools/train.py`：训练模型
2. 使用定制数据集进行训练
3. 类别不平衡数据集的重采样
4. 训练时应用图像的数据增强
5. `tools/analysis_tools/analyze_logs.py`：绘制学习曲线与评估训练速度
6. `tools/analysis_tools/optimize_anchors.py`：Anchor尺寸先验
7. `tools/analysis_tools/mot/browse_dataset.py`：浏览数据集

## （1）`tools/train.py`

`tools/train.py`可用于训练模型。

```python
# Single-gpu training
python tools/train.py \
    ${CONFIG_FILE} \
    [--work-dir ${WORK_DIR}] \ # 指定训练的工作目录
    [--resume ${CHECKPOINT_FILE}] \ # 从特定检查点恢复训练，若不指定则自动从work_dir中最新的检查点恢复
    [--cfg-options 'Key=value'] # 覆盖配置文件中的键值对

# CPU: disable GPUs and run single-gpu training script
export CUDA_VISIBLE_DEVICES=-1
python tools/train.py \
    ${CONFIG_FILE} \
    [--work-dir ${WORK_DIR}] \
    [--resume ${CHECKPOINT_FILE}]

# Multi-gpu training
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [--work-dir ${WORK_DIR}] \
    [--resume ${CHECKPOINT_FILE}]
```

### ⚪ 通过`--cfg-options`修改配置文件

训练时还可以通过`--cfg-options`传递配置文件键值对的修改信息，比如调整验证集上的评估间隔（默认为$1$ **epoch**）：

```python
python tools/train.py \
    ...
    --cfg-options model.train_cfg.val_interval=5
```

### ⚪ `--auto-scale-lr`学习率自动缩放

配置文件中的默认学习率适用于 $8$ 个 **GPU**、每个 **GPU** $2$ 个样本（**batch size** $16$）。通过设置学习率自动缩放，学习率将根据机器上的 **GPU** 数量和训练批量大小自动缩放。

```python
python tools/train.py \
    ...
    --auto-scale-lr \
```

## （2）使用定制数据集进行训练

**MMDetection** 推荐将数据集重新组织为 **COCO** 格式。

通过修改配置文件加载数据集：

```python
# 1. 修改数据集路径
data_root = '/data/zhengzj.zzj/wheat_head_counting/'
# 2. 修改类别名称
class_name = ('Wheat Head', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        # 3. 修改标注文件路径
        ann_file='train.json',
        # 4. 修改图像路径
        data_prefix=dict(img='train/')))
```

## （3）类别不平衡数据集的重采样
- 参考论文：[LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195)
- 代码位置：`./mmdet/datasets/dataset_wrappers.py`

对于**类别不平衡(class imbalanced)**的数据集，引入重采样的策略。即对于出现较少的类别对应的数据，将其重复采样多次用于扩充数据集；对于图像任务，每张图像采样的次数是由其**重复因子(repeat factor)**决定。图像$I$的重复因子$r(I)$定义为图像中最稀少的类别的重复因子:$r(I) = max_{c \in I} r(c)$；类别$c$的重复因子$r(c)$是由该类别在图像中出现的频率$f(c)$和人为给定的阈值$t$共同决定:$r(c) = \mathop{\max}(1, \sqrt{\frac{t}{f(c)}})$。

**mmdet**提供了**ClassBalancedDataset**类用于除了类别不平衡的数据集。设置参数`oversample_thr=1e-3`，出现频率低于该值的类别会被重复采样，重采样的程度由其重复因子决定。

如对数据集`Dataset_A`设置参数`oversample_thr=1e-3`，则修改**config**文件如下：

```python
### 修改前：
data = dict(
    ...
    train=dict(
        type='Dataset_A',
        ann_file='train.json',
        img_prefix='train-images/',
        pipeline=train_pipeline),
    ...)

### 修改后：
data = dict(
    ...
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ann_file='train.json',
            img_prefix='train-images/',
            pipeline=train_pipeline),
        ),
    ...)
```

## （4）训练时应用图像的数据增强
- 参考论文：[Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125)

使用[Albumentations](https://0809zheng.github.io/2021/06/15/albumentation.html)库为图像进行数据增强。**Albumentations**已经集成在**mmdetection**框架下。使用时直接修改`config`文件内的`train_pipeline`即可：

```python
albu_train_transforms = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='OneOf', transforms=[
            dict(type='Blur', blur_limit=3, p=0.5),
            dict(type='MedianBlur', blur_limit=3, p=0.5),
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Albu', transforms=albu_train_transforms),  # 数据增强
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
```

## （5）`tools/analysis_tools/analyze_logs.py`

`tools/analysis_tools/analyze_logs.py`根据给定的训练日志文件(**.json**格式)绘制损失曲线与**mAP**曲线。使用前需要安装依赖库：

```python
!pip install seaborn
```

### ⚪ 绘制学习曲线

```python
python tools/analysis_tools/analyze_logs.py plot_curve \
log.json \ # 读取训练日志，如果同时读取多个则并行写入：log1.json log2.json
[--keys ${KEYS}] \ # 损失类型：loss_cls loss_bbox bbox_mAP
[--eval-interval ${EVALUATION_INTERVAL}] \
[--title ${TITLE}] \
[--legend ${LEGEND}] \ # 图例注释，多个曲线则并行写入：run1 run2
[--backend ${BACKEND}] \
[--style ${STYLE}] \
[--out ${OUT_FILE}] # 图像存储位置：losses.pdf
```

### ⚪ 评估训练速度

```python
python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
```

## （6）`tools/analysis_tools/optimize_anchors.py`：Anchor尺寸先验

对于**anchor-based**的目标检测器，可以人为设置**Anchor**尺寸先验。

```python
python tools/analysis_tools/optimize_anchors.py \
    ${CONFIG} \
    --algorithm \ $ 方法 k-means 或  differential_evolution
    --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
    --output-dir ${OUTPUT_DIR} # 结果存储路径
```

## （7）`tools/analysis_tools/mot/browse_dataset.py`：浏览数据集

`tools/analysis_tools/mot/browse_dataset.py`可以浏览数据集，用于检查数据集标注是否正确：

```python
python tools/analysis_tools/browse_dataset.py \
    ${CONFIG_FILE} \
    [--show-interval ${SHOW_INTERVAL}]
```

# 2. 评估测试结果

1. `tools/test.py`：获取测试结果
2. `tta_model`和`tta_pipeline`：测试时增强
3. `tools/analysis_tools/analyze_results.py`：结果分析
4. `tools/analysis_tools/fusion_results.py`：多模型融合预测
5. `tools/analysis_tools/get_flops.py`：模型复杂度分析
6. `tools/analysis_tools/benchmark.py`：FPS分析
7. `tools/analysis_tools/confusion_matrix.py`绘制混淆矩阵

## （1）`tools/test.py`：获取测试结果

`tools/test.py`可用于获取测试结果。**MMDetection**中的测试集其实是验证集，需要在配置文件中同时指定图像路径与对应的标注文件路径（不妨设置与验证集相同）。

```python
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = val_evaluator
```

测试结果将会保存为**pickle**文件：

```python
# Single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \ # 输出pkl文件路径 results.pkl
    [--show] \ # 如果指定，检测结果将绘制在图像上并显示在新窗口中
    [--show-dir] \ # 如果指定，检测结果将绘制在图像上并保存到指定目录
    [--work-dir] \ # 如果指定，包含评估指标的检测结果将保存到指定目录
    [--cfg-options 'Key=value'] # 覆盖配置文件中的键值对

# CPU: disable GPUs and run single-gpu testing script
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# Multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}]
```

### ⚪ 通过`--cfg-options`修改配置文件

测试时还可以通过`--cfg-options`传递配置文件键值对的修改信息，比如测试时输出所有类别的**AP**值：

```python
./tools/dist_test.sh \
    ...
    --cfg-options test_evaluator.classwise=True
```

或者测试时进行批量图像推理：

```python
./tools/dist_test.sh \
    ...
    --cfg-options test_dataloader.batch_size=2
```

## （2）`tta_model`和`tta_pipeline`：测试时增强

**测试时增强（TTA）**是在测试阶段使用的数据增强策略。它对同一图像的不同增强版本（例如翻转和缩放）进行模型推理，然后平均每个增强图像的预测以获得更准确的结果。

使用 **TTA** 首先需要在配置文件中添加`tta_model`和`tta_pipeline`：

```python
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

img_scales = [(1333, 800), (666, 400), (2000, 1200)]
tta_pipeline = [
    dict(type='LoadImageFromFile',
        backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[ #  3 次多尺度变换
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [ #  2 次翻转变换（翻转和不翻转）
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict( # 使用 PackDetInputs 将图像打包成最终结果
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]
```

`--tta`在运行测试脚本时进行设置：

```python
# Single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--tta]
```

## （3）`tools/analysis_tools/analyze_results.py`：结果分析

`tools/analysis_tools/analyze_results.py`计算测试集中每张图像的**mAP**，并存储/展示得分最高/最低的**topK**张图像预测结果。


```python
python tools/analysis_tools/analyze_results.py \
      ${CONFIG} \          # 此处应指定测试配置文件路径！
      ${PREDICTION_PATH} \ # 输出pkl文件路径
      ${SHOW_DIR} \        # 预测结果存储路径
      [--show] \           # 是否展示图像
      [--wait-time ${WAIT_TIME}] \ # 图像展示间隔
      [--topk ${TOPK}] \   # 默认为20
      [--show-score-thr ${SHOW_SCORE_THR}] \ # 过滤得分较低的预测结果
      [--cfg-options ${CFG_OPTIONS}]
```

## （4）`tools/analysis_tools/fusion_results.py`：多模型融合预测

- 论文：[Weighted boxes fusion: Ensembling boxes from different object detection models](https://arxiv.org/abs/1910.13302)

`tools/analysis_tools/fusion_results.py`能够融合多个模型的检测结果，采用**Weighted Boxes Fusion(WBF)**方法。

**WBF**方法假设使用$N$个不同模型对相同的数据集进行边界框预测，然后做如下操作：
1. 将每个模型的每个预测边界框加入到一个单独的列表$\mathbf B$中，按照置信度$\mathbf C$进行递减排序；
2. 分别声明空列表$\mathbf L$和$\mathbf F$为边界框簇和融合后的边界框。列表$\mathbf L$中的每个位置可以包含一个边界框或者一个边界框集合，形成一个簇。$\mathbf F$中每个位置只包含一个边界框，对应从相应的$\mathbf L$中的簇中融合出来的边界框；
3. 在一个循环中迭代$\mathbf B$中的预测框，尝试找到列表$\mathbf F$中的一个匹配边界框。匹配的定义为边界框**IoU>**阈值，该实验阈值设置为$0.55$；
4. 如果找到匹配，则将该框添加到与列表$\mathbf F$中匹配框对应的列表$\mathbf L$中的$\mathbf{pos}$处；
5. 如果未找到匹配，则将$\mathbf B$作为一个新的实体加入到列表$\mathbf L$和$\mathbf F$的最后，继续对$\mathbf B$中下一个边界框进行处理；
6. 使用所有的在簇$\mathbf{L[pos]}$中的$T$个边界框加权计算$\mathbf{F[pos]}$中边界框的坐标和置信度得分（置信度越高的边界框对于融合边界框的坐标做出的贡献越大）：$C = \frac{\sum_t^TC_t}{T},X/Y=\frac{\sum_t^TC_t\cdot X_t/Y_t}{\sum_t^TC_t}$
7. 当$\mathbf B$中的所有边界框都被处理了，对$\mathbf F$列表的置信度得分重新估计：乘上簇中的边界框个数并除以模型数目（当簇中边界框数目少的时候，该融合框的置信度应该降低）：$C = C \cdot \frac{\min(T,N)}{N}$

**WBF**方法目前仅支持**COCO**格式的预测结果：

```python
python tools/analysis_tools/fuse_results.py \
       ${PRED_RESULTS} \ # 提供.json预测结果，多个预测结果并列
       [--annotation ${ANNOTATION}] \ # 提供标签文件
       [--weights ${WEIGHTS}] \ # 每个模型的预测权重
       [--fusion-iou-thr ${FUSION_IOU_THR}] \ # 边界框匹配阈值，默认0.55
       [--skip-box-thr ${SKIP_BOX_THR}] \ # 过滤得分较低的预测结果
       [--conf-type ${CONF_TYPE}] \ # 置信度加权方式：avg平均，max最大
       [--eval-single ${EVAL_SINGLE}] \ # 开启后，分别评估单个模型
       [--save-fusion-results ${SAVE_FUSION_RESULTS}] \ # 开启后，存储融合结果
       [--out-dir ${OUT_DIR}] # 存储结果的路径
```

## （5）`tools/analysis_tools/get_flops.py`：模型复杂度分析

`tools/analysis_tools/get_flops.py`能够分析模型的**FLOPs**和参数量。

```python
python tools/analysis_tools/get_flops.py \
    ${CONFIG_FILE} \
    [--shape ${INPUT_SHAPE}] # 格式：H W
```

一些常见结论：
- 默认输入尺寸为$(1, 3, 1280, 800)$
- **FLOPs**与输入尺寸相关，参数量与输入尺寸无关
- 一些操作（如**GN**或自定义操作）没有计入**FLOPs**
- 两阶段检测器的**FLOPs**与**proposal**数量有关

## （6）`tools/analysis_tools/benchmark.py`：FPS分析

`tools/analysis_tools/benchmark.py`能够计算**FPS**，即网络每秒处理的图像帧数（包括前向传播与后处理过程）。通常只使用单**GPU**配置以获得准确的**FPS**估计。

```python
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT} \
    tools/analysis_tools/benchmark.py \
    ${CONFIG} \
    [--checkpoint ${CHECKPOINT}] \
    [--repeat-num ${REPEAT_NUM}] \
    [--max-iter ${MAX_ITER}] \
    [--log-interval ${LOG_INTERVAL}] \
    --launcher pytorch
```

## （7）`tools/analysis_tools/confusion_matrix.py`绘制混淆矩阵

`tools/analysis_tools/confusion_matrix.py`能够根据预测的**.pkl**文件绘制不同类别之间的混淆矩阵。

```python
python tools/analysis_tools/confusion_matrix.py \
    ${CONFIG} \
    ${DETECTION_RESULTS} \
    ${SAVE_DIR} --show
```