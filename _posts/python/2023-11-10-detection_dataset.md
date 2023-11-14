---
layout: post
title: '目标检测数据集的分析'
date: 2023-11-10
author: 郑之杰
cover: ''
tags: Python
---

> Analysis on Object Detection Datasets.

目标检测数据集需要标记各个目标的位置和类别。一般的目标区域位置用一个矩形框来表示，称之为边界框（**bounding box, bbox**），常用以下$3$种方式表达：
- $(x_1,y_1,x_2,y_2)$：$(x_1,y_1)$为左上角坐标，$(x_2,y_2)$为右下角坐标；
- $(x_1,y_1,w,h)$：$(x_1,y_1)$为左上角坐标，$w$为边界框高度，$h$为边界框宽度；
- $(x_c,y_c,w,h)$：$(x_c,y_c)$为边界框中心坐标，$w$为边界框高度，$h$为边界框宽度。

其中常见的目标检测数据集如**Pascal VOC**采用$(x_1,y_1,x_2,y_2)$表示物体的边界框, **COCO**采用$(x_1,y_1,w,h)$表示物体的边界框。因此通常也称为**VOC**格式和**COCO**格式。

# 1. 目标检测数据集的格式分析

## （1）VOC格式

**VOC**格式中，每个图像文件对应一个同名的**xml**文件，**xml**文件内包含对应图片的基本信息，比如文件名、来源、图像尺寸以及图像中包含的物体区域信息和类别信息等。

一个典型的**xml**文件如下：

```xml
<annotation>
	<folder>IP103_final_new1</folder>
	<filename>IP000000000.jpg</filename>
	<path>/home/ubuntu2/Desktop/IP103_final_new1/IP000000000.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>650</width>
		<height>420</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>0</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>99</xmin>
			<ymin>231</ymin>
			<xmax>524</xmax>
			<ymax>334</ymax>
		</bndbox>
	</object>
</annotation>
```

**xml**文件中通常包含以下关键字段：
- `filename`：表示图像名称
- `size`：表示图像尺寸。包括：图像宽度、图像高度、图像深度
- `object`：表示每个物体，包括：
1. `name`：物体类别名称
2. `pose`：关于目标物体姿态描述（非必须字段）
3. `truncated`：如果物体的遮挡超过$15-20\％$并且位于边界框之外，请标记为**truncated**（非必须字段）
4. `difficult`：难以识别的物体标记为**difficult**（非必须字段）
5. `bndbox`：**(xmin,ymin)** 左上角坐标，**(xmax,ymax)** 右下角坐标

## （2）COCO格式

**COCO**格式中，数据标注是将所有训练图像的标注都存放到一个**json**文件中。数据以字典嵌套的形式存放。

可以通过**json**文件的读取获取**COCO**格式标注文件的部分信息：

```python
# 查看COCO标注文件
import json
coco_anno = json.load(open('train.json'))

# coco_anno.keys
print('keys:', coco_anno.keys())

# 查看类别信息
print('\n物体类别:', coco_anno['categories'])

# 查看一共多少张图
print('\n图像数量：', len(coco_anno['images']))

# 查看一张图像信息
print('\n图像信息：', coco_anno['images'][0])

# 查看一共多少个目标物体
print('\n标注物体数量：', len(coco_anno['annotations']))

# 查看一条目标物体标注信息
print('\n查看一条目标物体标注信息：', coco_anno['annotations'][0])
```

```python
keys: dict_keys(['images', 'type', 'annotations', 'categories', 'info'])

物体类别: [{'supercategory': 'none', 'id': 1, 'name': 'Wheat Head'}]

图像数量： 6515

图像信息： {'file_name': '4563856cc6d75c670eafd86d5eb7245fbe8f273c28f9e36f7c6aaf097c7ce423.png', 'height': 512, 'width': 512, 'id': 2}

标注物体数量： 275466

查看一条目标物体标注信息： {'area': 1184, 'iscrowd': 0, 'bbox': [49, 346, 32, 37], 'category_id': 1, 'ignore': 0, 'segmentation': [], 'image_id': 2, 'id': 0}
```

**COCO**格式中比较关键的**key**包括：
- `images`：表示标注文件中图像信息列表，每个元素是一张图像的信息。
- `annotations`：表示标注文件中目标物体的标注信息列表，每个元素是一个目标物体的标注信息。
- `categories`：表示标注文件中所有的类别及其对应的索引。

## （3）VOC格式转COCO格式

**VOC**格式下，有多少张图像就有多少个标注信息，因此数据越多，读取的也就越慢；而**COCO**格式只有一个标注文件，不管数据量有多么地庞大，只需要读取一个文件就可以了。因此**VOC**格式转**COCO**格式可以提高文件读取的效率。

```python
import xml.etree.ElementTree as ET
import os
import json
 
coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
 
category_set = dict()
image_set = set()
 
category_item_id = -1
image_id = 0
annotation_id = 0
 
def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id
 
def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id
 
def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
 
    annotation_item['segmentation'].append(seg)
 
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)
 
def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids
 
 
"""直接从xml文件夹中生成"""
def parseXmlFiles(xml_path, json_save_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
 
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
 
        xml_file = os.path.join(xml_path, f)
        print(xml_file)
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
 
        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
 
            if elem.tag == 'folder':
                continue
 
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
 
            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None
 
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]
 
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)
 
                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)
 
                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                                                                   bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
    json.dump(coco, open(json_save_path, 'w'))
 
 
if __name__ == '__main__':
    ann_path = "Annotations"
    json_save_path = "annotations.json"
    parseXmlFiles(ann_path, json_save_path)
```

## （4）COCO格式转VOC格式

**COCO**格式无法直接被数据标注软件直接读取，想要检查数据中是否有脏数据（错标注、误标注和漏标注），最直观的方法就是把他们可视化出来，将**COCO**格式转**VOC**格式可以方便我们对数据集里的标注信息进行修改；另一方面，如果想要添加新数据，像**Labellmg**这样的标注工具导出的标注格式通常都是**VOC**格式的。

```python
from pycocotools.coco import COCO
import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
    
# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def save_annotations(filename, objs, filepath):
    annopath = CKanno_dir + "/" + filename[:-3] + "xml"  # 生成的xml文件保存路径
    dst_path = CKimg_dir + "/" + filename
    img = cv2.imread(filepath)
    shutil.copy(filepath, dst_path)  # 把原始图像复制到目标文件夹
    
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)

def handle_per_img(coco, img, classes, origin_image_dir):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir,  filename)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
    save_annotations(filename, objs, filepath)

def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes

def COCO2VOC(origin_anno_dir, origin_image_dir):
    coco = COCO(origin_anno_dir)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        handle_per_img(coco, img, classes, origin_image_dir)
            

if __name__ == "__main__":
    CKimg_dir = 'JPEGImages_A'   # 1、生成图片保存的路径
    CKanno_dir = 'Annotations_A' # 2、生成标注文件保存的路径
    mkr(CKimg_dir)
    mkr(CKanno_dir)
    
    origin_image_dir = 'JPEGImages'       # 3、原始的coco的图像存放位置
    origin_anno_dir = 'annotations.json'  # 4、原始的coco的标注存放位置
    COCO2VOC(origin_anno_dir, origin_image_dir)
```


# 2. 目标检测数据集的统计分析

## （1）各类别实例数量分析

可以统计一下数据集里每个类别的数量都有多少。如果有些类别的实例数量比较少，可以考虑使用数据增强等方法缓解数据不均衡的问题。

### ⚪ VOC格式

```python
import os
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

def count_num(indir):
    # 提取xml文件列表
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    dict = defaultdict(int) # 新建字典用于存放各类标签及其对应的数目
    
    # 遍历xml文件
    for i, file in tqdm(enumerate(annotations)):
        in_file = open(file, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        # 遍历文件的所有标签
        for obj in root.iter('object'):
            name = obj.find('name').text
            dict[name] += 1
    
    # 打印结果
    print('各类标签的数量分别为：')
    for key in dict.keys():
        print(key + ':' + str(dict[key]))

indir = './Annotations'
count_num(indir)
```

### ⚪ COCO格式

```python
import json
from tqdm import tqdm
from collections import defaultdict

coco_anno = json.load(open('annotations.json'))
dict = defaultdict(int) # 新建字典用于存放各类标签及其对应的数目
for anno in tqdm(coco_anno['annotations']):
    dict[anno['category_id']] += 1

# 建立目标id与真实类别的映射
classes = defaultdict(str)
for cate in coco_anno['categories']:
    classes[cate['id']] = cate['name']

# 打印结果
print('各类标签的数量分别为：')
for key in dict.keys():
    print(classes[key] + ':' + str(dict[key]))
```

## （2）检测框高宽比分析

高宽比的设定在目标检测中是很重要的参数。可以画一个检测框高宽比分布直方图，从而反应当前检测框款高宽比的分布情况。


### ⚪ VOC格式

```python
import os
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def ratio(indir):
    # 提取xml文件列表
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    count = []
    
    # 遍历xml文件
    for i, file in tqdm(enumerate(annotations)):
        in_file = open(file, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        # 遍历文件的所有检测框
        for obj in root.iter('object'):
            xmin = obj.find('bndbox').find('xmin').text
            ymin = obj.find('bndbox').find('ymin').text
            xmax = obj.find('bndbox').find('xmax').text
            ymax = obj.find('bndbox').find('ymax').text
            if int(xmax)-int(xmin) != 0:
                aspect_ratio = (int(ymax)-int(ymin)) / (int(xmax)-int(xmin))
            count.append(aspect_ratio)
        
    # 绘制高宽比的直方图
    plt.hist(count, bins=20)
    plt.show()
            
indir  = './Annotations'
ratio(indir) 
```

### ⚪ COCO格式

```python
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

coco_anno = json.load(open('annotations.json'))
count = []
for anno in tqdm(coco_anno['annotations']):
    _, _, w, h = anno['bbox']
    count.append(h/w)

# 绘制高宽比的直方图
plt.hist(count, bins=20)
plt.show()
```

## （3）图像尺寸分析

数据集中并不是所有图像的尺寸都是固定的，可以进行图像尺寸的分析。

### ⚪ VOC格式

```python
import os
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

def image_size(indir):
    # 提取xml文件列表
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    width_heights = set()
    
    # 遍历xml文件
    for i, file in tqdm(enumerate(annotations)):
        in_file = open(file, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        width_heights.add((width, height))
        
    print('数据集中共有{}种不同的尺寸，分别是：'.format(len(width_heights)))
    for item in width_heights:
        print(item)

indir = './Annotations'
image_size(indir)  
```

### ⚪ COCO格式

```python
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

coco_anno = json.load(open('annotations.json'))
width_heights = set()
for image in tqdm(coco_anno['images']):
    width_heights.add((image['width'], image['height']))

print('数据集中共有{}种不同的尺寸，分别是：'.format(len(width_heights)))
for item in width_heights:
    print(item)
```


## （4）检测框中心分布分析

可以画一个检测框中心分布散点图，直观地反应检测框中心点在图像中的位置分布。

### ⚪ VOC格式

```python
import os
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

def distribution(indir):
    # 提取xml文件列表
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.xml')
    data_x, data_y = [], []
    
    # 遍历xml文件
    for i, file in tqdm(enumerate(annotations)):
        in_file = open(file, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        # 遍历文件的所有检测框
        for obj in root.iter('object'):
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text) 
            xc = (xmin + (xmax-xmin)/2) / width
            yc = (ymin + (ymax-ymin)/2) / height
            data_x.append(xc)
            data_y.append(yc)
    
    plt.scatter(data_x, data_y, s=1, alpha=.1)

indir = './Annotations'
distribution(indir)
```

### ⚪ COCO格式

```python
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

coco_anno = json.load(open('annotations.json'))
image_size = {}         # 记录图像尺寸
data_x, data_y = [], [] # 记录检测框中心位置
for image in coco_anno['images']:
    image_size[image['id']] = (image['width'], image['height'])

for anno in tqdm(coco_anno['annotations']):
    width, height = image_size[anno['image_id']]
    x1, y1, w, h = anno['bbox']
    xc = (x1 + w/2) / width
    yc = (y1 + h/2) / height
    if 0 <= xc <= 1 and 0 <= yc <= 1:
        data_x.append(xc)
        data_y.append(yc)
    
plt.scatter(data_x, data_y, s=1, alpha=.1)   
```