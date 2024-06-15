import yaml
import cv2
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from datasets.factory import Datasets

from trackers.frog_kcf import FrogKCFTracker

from simple_fovea.fovea_optimize import FoveaOptimizer
from simple_fovea.fovea_obj_detect import Fovea_FRCNN_FPN, get_processed_boxes
from simple_fovea.box_iou import box_iou

if __name__ == '__main__':

    with open('experiments/cfgs/test.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dataset = config["dataset"]
    frame_range = config["frame_range"]

    datasets = Datasets(dataset)

    # 中央凹设置

    # 中央凹区域大小在原始图像中的比例
    fovea_scale_width = config["fovea_scale"]["width"]
    fovea_scale_height = config["fovea_scale"]["height"]

    # 处理后图像相较于原始图像的压缩比
    compress_ratio_width = config["compress_ratio"]["width"]
    compress_ratio_height = config["compress_ratio"]["height"]

    # 中央凹区域的检测模型
    fovea_obj_detect = Fovea_FRCNN_FPN(num_classes=2)
    obj_detect_weight_dir = config["fovea_obj_detect_weight"]
    obj_detect_state_dict = torch.load(obj_detect_weight_dir)
    fovea_obj_detect.load_state_dict(obj_detect_state_dict)
    if torch.cuda.is_available():
        fovea_obj_detect = fovea_obj_detect.cuda()


    # PIL to PyTorch向量的变换
    transforms = ToTensor()

    for seq in datasets:
        print(f'Found seq: {seq}')

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))
        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))

        active_kcf_trackers = []

        fovea_width = -1
        fovea_height = -1
        origin_img_width = -1
        origin_img_height = -1
        fovea_optimizer = None

        for index, frame_data in enumerate(seq_loader):

            if index % 50 == 0:
                print(f"Processing No.{index} image\n")
            
            # 获取压缩过后的图片
            img = frame_data['img']
            img_path = frame_data['img_path']

            # 读取cv2的格式
            cv2_img = cv2.imread(img_path)

            # 如果是第一帧，那么根据第一帧的原始图像大小初始化中央凹优化器

            if fovea_width == -1:
                # 加载原始图像
                origin_img_path = frame_data['origin_img_path']
                # 用PIL读取原始图像
                origin_image = Image.open(origin_img_path).convert('RGB')
                origin_image = transforms(origin_image)
                origin_img_width, origin_img_height = origin_image.shape[2], origin_image.shape[1]
                # 根据原始图像宽高计算中央凹区域大小
                fovea_width = int(origin_img_width * fovea_scale_width)
                fovea_height = int(origin_img_height * fovea_scale_height)

                print(f"Initialize Fovea. Image Size: {origin_img_width}x{origin_img_height}, Fovea Size: {fovea_width}x{fovea_height}")
                
                # 初始化中央凹优化器
                fovea_optimizer = FoveaOptimizer(img_width=origin_img_width, img_height=origin_img_height, init_image_path=None,
                                                 region_scale=0.025, pixel_change_threshold=70,
                                                 fovea_width=fovea_width, fovea_height=fovea_height,
                                                 is_PIL=True)
            else:
                # 已经初始化过了，只需要读取图像
                origin_image = Image.open(origin_img_path).convert('RGB')
                origin_image = transforms(origin_image)

            # 记录上一帧中所有活跃的锚框
            prev_online_boxes = []

            # 让所有已有的KCF追踪器在img上跑一遍

            for kcf_tracker in active_kcf_trackers:
                # 中央凹区域是基于上一帧的追踪结果的，所以这里先加bbox再更新
                prev_online_boxes.append(kcf_tracker.current_track['bbox'])
                track_id, ok, bbox = kcf_tracker.update_tracker(cv2_img, index)
                if not ok:
                    print(f'Track {track_id} failed')
                    active_kcf_trackers.remove(kcf_tracker)
                
            
            # 选择中央凹区域
            fovea_x, fovea_y = fovea_optimizer.get_fovea_position(current_frame_img=img[0],
                                                                  prev_online_boxes=prev_online_boxes,
                                                                  visualize=False)
            
            # 处理边界情况
            if np.isnan(fovea_x) or np.isnan(fovea_y):
                fovea_x = int(origin_img_width / compress_ratio_width / 4)
                fovea_y = int(origin_img_height / compress_ratio_height / 4)
            
            # 放大回原始图像的尺寸
            fovea_x = int(fovea_x * compress_ratio_width)
            fovea_y = int(fovea_y * compress_ratio_height)

            # 得到中央凹区域的tlwh
            fovea_pos = (fovea_x, fovea_y, fovea_width, fovea_height)

            # 截取出给定中央凹区域tlwh对应的原始图像内容
            fovea_img = origin_image[:, fovea_y:fovea_y + fovea_height, fovea_x:fovea_x + fovea_width]
            _fovea_img = torch.stack([fovea_img], dim=0)

            # 送入中央凹区域的目标检测器进行检测，获得该目标检测器输出的boxes和scores
            fovea_boxes, fovea_scores = fovea_obj_detect.detect(_fovea_img)

            # 将所有中央凹区域得到的目标boxes转换到经过压缩的图像的坐标下
            processed_fovea_boxes = get_processed_boxes(fovea_boxes=fovea_boxes, fovea_pos=fovea_pos,
                                                        compress_ratio=[compress_ratio_width, compress_ratio_height])
            
            # 对每个processed_fovea_boxes，检查其与所有当前活跃的KCF追踪器的IOU，如果IOU大于0.5，那么认为是同一个目标
            # 不加以处理；否则，新建一个KCF追踪器，并赋予新的id
            
            # 先将所有的active tracker的bbox合并起来
            active_kcf_tracker_bboxes = []
            for kcf_tracker in active_kcf_trackers:
                tracker_bbox = kcf_tracker.current_track['bbox']
                active_kcf_tracker_bboxes.append(tracker_bbox)
            
            # 逐个计算iou
            for fovea_box in processed_fovea_boxes:
                ious = box_iou(fovea_box, active_kcf_tracker_bboxes)
                if (ious < 0.5).all():
                    # 新建一个KCF追踪器
                    new_kcf_tracker = FrogKCFTracker(index)
                    # 将本帧（压缩后的）内容送入KCF追踪器做初始化
                    new_kcf_tracker.init_tracker(frame=cv2_img, bbox=fovea_box)
                    active_kcf_trackers.append(new_kcf_tracker)
            


