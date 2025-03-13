import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from typing import List, Dict

def parse_halcon_annotation(ann_path: str) -> List[Dict]:
    """
    该部分解析Halcon标注文件
    假设Halcon标注文件格式：
    <annotation>
        <object>
            <name>class_name</name>
            <bndbox>
                <x1>100</x1>
                <y1>200</y1>
                <x2>300</x2>
                <y2>400</y2>
            </bndbox>
        </object>
        ...
    </annotation>
    """
    tree = ET.parse(ann_path)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('object'):
        obj_info = {
            'name': obj.find('name').text,
            'bndbox': {
                'x1': int(obj.find('bndbox/x1').text),
                'y1': int(obj.find('bndbox/y1').text),
                'x2': int(obj.find('bndbox/x2').text),
                'y2': int(obj.find('bndbox/y2').text)
            }
        }
        annotations.append(obj_info)
    return annotations

def convert_to_voc(
    img_dir: str,
    ann_dir: str,
    output_dir: str,
    class_names: List[str]
):

    # 创建VOC目录结构
    os.makedirs(os.path.join(output_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ImageSets', 'Main'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'JPEGImages'), exist_ok=True)

    image_ids = []
    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # 基础信息
        img_id = os.path.splitext(img_file)[0]
        image_ids.append(img_id)
        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, f"{img_id}.xml")
        
        # 读取图像尺寸
        with Image.open(img_path) as img:
            width, height = img.size
        
        # 创建XML结构
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'VOC'
        ET.SubElement(root, 'filename').text = img_file
        ET.SubElement(root, 'path').text = img_path
        
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'
        
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'
        
        # 解析标注
        annotations = parse_halcon_annotation(ann_path)
        
        for ann in annotations:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = ann['name']
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(ann['bndbox']['x1'])
            ET.SubElement(bndbox, 'ymin').text = str(ann['bndbox']['y1'])
            ET.SubElement(bndbox, 'xmax').text = str(ann['bndbox']['x2'])
            ET.SubElement(bndbox, 'ymax').text = str(ann['bndbox']['y2'])
        
        # 保存XML文件
        xml_str = ET.tostring(root, 'utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
        with open(os.path.join(output_dir, 'Annotations', f"{img_id}.xml"), 'w') as f:
            f.write(pretty_xml)
        
        # 复制图片（实际使用时建议直接链接或手动复制）
        # shutil.copy(img_path, os.path.join(output_dir, 'JPEGImages', img_file))

    # 创建ImageSets
    with open(os.path.join(output_dir, 'ImageSets/Main/trainval.txt'), 'w') as f:
        f.write('\n'.join(image_ids))

def convert_to_coco(
    img_dir: str,
    ann_dir: str,
    output_path: str,
    class_names: List[str]
):
    """
    转换为COCO格式数据集
    """
    coco = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 创建类别信息
    for i, name in enumerate(class_names, 1):
        coco['categories'].append({
            "id": i,
            "name": name,
            "supercategory": "none"
        })
    
    # 建立类别名称到ID的映射
    category_ids = {name: i for i, name in enumerate(class_names, 1)}
    
    ann_id = 1
    for img_id, img_file in enumerate(os.listdir(img_dir), 1):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # 图像信息
        img_path = os.path.join(img_dir, img_file)
        with Image.open(img_path) as img:
            width, height = img.size
        
        image_info = {
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height,
            "date_captured": "2023-01-01",
            "license": 0,
            "coco_url": "",
            "flickr_url": ""
        }
        coco['images'].append(image_info)
        
        # 解析标注
        ann_path = os.path.join(ann_dir, f"{os.path.splitext(img_file)[0]}.xml")
        annotations = parse_halcon_annotation(ann_path)
        
        for ann in annotations:
            x1 = ann['bndbox']['x1']
            y1 = ann['bndbox']['y1']
            x2 = ann['bndbox']['x2']
            y2 = ann['bndbox']['y2']
            
            width = x2 - x1
            height = y2 - y1
            
            coco['annotations'].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_ids[ann['name']],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1
    
    # 保存COCO格式
    with open(output_path, 'w') as f:
        json.dump(coco, f)

if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    HALCON_IMG_DIR = "/path/to/halcon/images"
    HALCON_ANN_DIR = "/path/to/halcon/annotations"
    VOC_OUTPUT_DIR = "/path/to/voc_dataset"
    COCO_OUTPUT_PATH = "/path/to/coco/annotations.json"
    CLASS_NAMES = ["cat", "dog", "person"]  # 根据实际类别修改
    
    # 执行转换
    convert_to_voc(HALCON_IMG_DIR, HALCON_ANN_DIR, VOC_OUTPUT_DIR, CLASS_NAMES)
    convert_to_coco(HALCON_IMG_DIR, HALCON_ANN_DIR, COCO_OUTPUT_PATH, CLASS_NAMES)
