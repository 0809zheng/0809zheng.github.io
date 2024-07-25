---
layout: post
title: '微调 Grounding DINO 和 Label Studio 进行半自动化目标检测标注'
date: 2024-01-19
author: 郑之杰
cover: 'https://pic.imgdb.cn/item/65b9f1c1871b83018ae5a80b.jpg'
tags: Python
---

> Semiautomatic Image Annotation with Grounding DINO and Label Studio.

## 0. 前言

[**Grounding DINO**](https://0809zheng.github.io/2023/11/02/groundingdino.html)是一种强大的开集目标检测器，能够根据用户指定的任意类别名称进行目标检测。尽管**Label Studio**官方提供了**Grounding DINO** 和 **Label Studio**结合的[**DEMO**](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/grounding_dino)用于进行开集目标检测的半自动标注，但是目前官方**Grounding DINO**模型并未开源相关的微调接口，从而在面对具体下游任务进行微调**Grounding DINO**、并进行半自动标注的需求时比较困难。

**OpenMMLab**社区开源了可以针对**Grounding DINO**模型进行训练和微调的[**MM Grounding DINO**](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md)项目。本项目建立在该项目的基础上，既可以直接使用**MM Grounding DINO**开源的预训练模型进行开集检测的半自动标注工作，也可以针对定制的数据集微调模型之后用于检测标注。

## 1. 环境配置

首先需要创建一个虚拟环境，然后安装 **PyTorch** 和 **MMCV**。接下来安装 **MMDetection**、**Label-Studio** 和 **label-studio-ml-backend**，具体步骤如下：

创建虚拟环境：

```python
conda create -n mmdet python=3.9 -y
conda activate mmdet
```

安装 **PyTorch**：

```python
# Linux and Windows CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Linux and Windows CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

安装 **MMCV**：

```python
pip install -U openmim
mim install "mmcv>=2.0.0"
# 安装 mmcv 的过程中会自动安装 mmengine
```

安装 **MMDetection**及额外的依赖包：

```python
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
pip install -v -e .
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git"
```

安装 **Label-Studio** 和 **label-studio-ml-backend**：

```python
# 安装 label-studio 需要一段时间,如果找不到版本请使用官方源
conda install -c anaconda libpq # 不运行则安装label-studio报错
pip install label-studio==1.7.2
pip install label-studio-ml==1.0.9
```

下载**MM Grounding DINO**模型开源的权重：

```python
mkdir workdirs
cd workdirs
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
cd ..
```

最后需要安装在 **Label Studio** 中使用**Grounding DINO**的**Label Studio ML SDK**模板：

```python
git clone https://github.com/0809zheng/mm_groundingdino
```

## 2. 启动服务

启动 **MM Grounding DINO** 后端推理服务：

```python
label-studio-ml start mm_groundingdino/run_without_docker --with \
config_file=./configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py \
checkpoint_file=./workdirs/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
device=cpu \
--port 8003
# device=cpu 为使用 CPU 推理，如果使用 GPU 推理，device=cuda:0 
```

![](https://pic.imgdb.cn/item/65aa3b8b871b83018a504acc.jpg)

此时，**MM Grounding DINO** 后端推理服务已经启动，后续在 **Label-Studio Web** 系统中配置 **http://localhost:8003** 后端推理服务即可。

现在启动 **Label-Studio** 网页服务：

```python
label-studio start
```

打开浏览器访问 **http://localhost:8080/** 即可看到 **Label-Studio** 的界面。

![](https://pic.imgdb.cn/item/65aa3c0b871b83018a521117.jpg)

## 3. 配置服务

注册一个**Label-Studio**用户，然后创建一个 **GroundingDINO-Semiautomatic-Label** 项目。

![](https://pic.imgdb.cn/item/65aa3cb9871b83018a547863.jpg)

点击 **Data Import** 导入需要标注的图片。

![](https://pic.imgdb.cn/item/65aa3cdf871b83018a54f43b.jpg)

然后选择 **Object Detection With Bounding Boxes** 模板（**Settings -> Labeling Interface -> Browse Templates**）：

![](https://pic.imgdb.cn/item/65aa3d4c871b83018a566300.jpg)

然后将感兴趣的检测类别添加到 **Label-Studio**，然后点击 **Save**。

![](https://pic.imgdb.cn/item/65aa3d94871b83018a575790.jpg)

然后在设置中点击 **Add Model** 添加 **MM Grounding DINO** 后端推理服务（**Settings -> Machine Learning**）。

![](https://pic.imgdb.cn/item/65aa3de8871b83018a587693.jpg)

输入**MM Grounding DINO** 后端推理服务配置 **http://localhost:8003**，打开 **Use for interactive preannotations**，然后点击 **Validate and Save**。

![](https://pic.imgdb.cn/item/65aa3e75871b83018a5a792f.jpg)

看到如下 **Connected** 就说明后端推理服务添加成功。

![](https://pic.imgdb.cn/item/65aa439f871b83018a6c5e95.jpg)

若报错：

```python
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like bert-base-uncased is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

这是因为**MM Grounding DINO** 采用了 **BERT** 作为语言模型，需要访问 **https://huggingface.co/**, 如果因为网络访问问题遇到连接错误，可以在有网络访问权限的电脑上下载所需文件（[**https://huggingface.co/bert-base-uncased**](https://huggingface.co/bert-base-uncased)）并保存在本地。

![](https://pic.imgdb.cn/item/65aa4184871b83018a6565b5.jpg)

最后，修改配置文件（`./configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py`）中的 **lang_model_name** 字段为本地路径即可。具体请参考以下代码：

```python
# lang_model_name = 'bert-base-uncased'
lang_model_name = '/your/path/to/bert-base-uncased'
```

## 4. 开始半自动化标注

选中图片，点击 **Label All Task** 开始标注：

![](https://pic.imgdb.cn/item/65aa43c3871b83018a6cd7db.jpg)

可以看到 **MM Grounding DINO** 后端推理服务已经成功返回了预测结果并显示在图片上；可以手工拖动框，修正一下框的位置，然后点击 **Submit**，本张图片就标注完毕了。

![](https://pic.imgdb.cn/item/65add30c871b83018ac929bb.jpg)

**submit** 完毕所有图片后，点击 **exprot** 导出 **COCO** 格式的数据集，就能把标注好的数据集的压缩包导出来了。

![](https://pic.imgdb.cn/item/65add410871b83018acba551.jpg)

打开解压后的文件夹，可以看到标注好的数据集，包含了图片和 **json** 格式的标注文件。

![](https://pic.imgdb.cn/item/65add49c871b83018acd1520.jpg)

如果想在 **Label-Studio** 中使用微调后的 **Grounding DINO**模型，可以参考在启动后端推理服务时，将 **config_file** 和 **checkpoint_file** 替换准备好的配置文件和权重文件即可，例如：

```python
cd path/to/mmetection

label-studio-ml start projects/LabelStudio/backend_template --with \
config_file=path/to/your/groundingdino_config.py \
checkpoint_file=path/to/your/groundingdino_weights.pth \
device=cpu \
--port 8003
# device=cpu 为使用 CPU 推理，如果使用 GPU 推理，device=cuda:0 
```

## 5. 设置Docker镜像

通过 **Docker** 镜像可以把上述应用程序和配置依赖打包好形成一个可交付的运行环境，从而可以在任何物理设备上运行，并将其连接到 **Label Studio** 以执行半自动标注任务。

首先确保环境中已安装**Docker**和 **Label Studio ML SDK**模板：

```python
cd mmdetection/mm_groundingdino
```

该程序的目录结构如下所示：

```
 mm_groundingdino/
 ├── _wsgi.py
 ├── docker-compose.yml
 ├── Dockerfile
 ├── io.py
 ├── mmdetection.py
 ├── README.md
 └── requirements.txt
```

`_wsgi.py`是使用 **Docker** 运行 **ML** 后端的帮助程序文件（通常无需修改）。`docker-compose.yml`是配置镜像参数的文件。`Dockerfile`是给出**Docker**执行命令的主文件。`requirements.txt`是给出模型预测过程的主文件。`requirements.txt`是列出**Python**依赖项的文件。`io.py`是调整**label studio**中图片下载逻辑的辅助文件文件。

将预先下载的**BERT** 模型（`'/your/path/to/bert-base-uncased'`）复制到路径`mm_groundingdino/`下。同时将预训练或微调后的**Grounding DINO**模型的 配置文件**config_file** 和 权重文件**checkpoint_file** 复制到路径`mm_groundingdino/`下。

修改`Dockerfile`中关于配置文件和权重文件的指令：
```python
COPY grounding_dino_swin-t_finetune_8xb4_20e_cat.py .
COPY best_coco_bbox_mAP_epoch_14.pth .

ENV checkpoint_file=/app/best_coco_bbox_mAP_epoch_14.pth
ENV config_file=/app/grounding_dino_swin-t_finetune_8xb4_20e_cat.py
```

修改`docker-compose.yml`中关于**Label Studio**的配置：

```python
environment:
    # Add these variables if you want to access the images stored in Label Studio
    - LABEL_STUDIO_HOST=http://121.41.23.189:8082 # Change to http://localhost:8080/
    - LABEL_STUDIO_ACCESS_TOKEN=267559dbedb55009e421e8e8733327297a6afefb
```

其中`LABEL_STUDIO_ACCESS_TOKEN`可通过**Account & Settings -> Access Token**查询。

修改`docker-compose.yml`中关于端口号的设置：

```python
ports:
    - 9090:9090 # Change to XXXX:9090
```

准备工作完成之后，安装镜像（安装速度与网速有关）：

```python
docker build -t mm_groundingdino_server:v01 .
```
![](https://pic.imgdb.cn/item/65b9f10e871b83018ae343a3.jpg)

安装完成之后，启动镜像：

```python
docker-compose up # -d 表示后端启动
```

![](https://pic.imgdb.cn/item/65b9f13b871b83018ae3e2d8.jpg)

之后在**Label Studio**中配置后端推理服务 **http://localhost:9090** 即可。

### ⚪ 参考文献
- [使用 MMDETECTION 和 LABEL-STUDIO 进行半自动化目标检测标注](https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/label_studio.html)
- [Using Text Prompts for Image Annotation with Grounding DINO and Label Studio](https://labelstud.io/blog/using-text-prompts-for-image-annotation-with-grounding-dino-and-label-studio/)
- [label-studio-ml-backend: Configs and boilerplates for Label Studio's Machine Learning backend](https://github.com/HumanSignal/label-studio-ml-backend/tree/master)
