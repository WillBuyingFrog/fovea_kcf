import yaml
import os
import random
import cv2
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

from datasets.factory import Datasets

from trackers.frog_kcf import FrogKCFTracker

from simple_fovea.fovea_optimize import FoveaOptimizer
from simple_fovea.fovea_obj_detect import Fovea_FRCNN_FPN, get_processed_boxes
from simple_fovea.box_iou import box_iou_numpy
from simple_fovea.bbox_utils import tlbr_to_tlwh_numpy, tlwh_to_tlbr_numpy
from simple_fovea.visualize import visualize_all

if __name__ == '__main__':

    seed = 12345

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    with open('experiments/cfgs/test.yaml', 'r') as file:
        config = yaml.safe_load(file)

    exp_name = config["name"]
    visualize_output_root = config["visualize_output_root"]
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
    fovea_obj_detect.eval()
    if torch.cuda.is_available():
        fovea_obj_detect = fovea_obj_detect.cuda()

    print(f"Loaded Fovea Object Detector from {obj_detect_weight_dir}\n")

    # 检测模型的行人检测置信度阈值
    detect_person_thresh = config["detect_person_thresh"]

    # PIL to PyTorch向量的变换
    transforms = ToTensor()

    # 追踪目标计数器
    id_counter = 0

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
        compressed_img_width = -1
        compressed_img_height = -1
        fovea_optimizer = None

        # 保存到 visualize_output_root/name/seq_name/
        visualize_output_path = f"{visualize_output_root}/{exp_name}/{seq}"
        # 如果这个path还没有，就新建
        if not os.path.exists(visualize_output_path):
            os.makedirs(visualize_output_path)
        else:
            # 否则先清空里面已有的所有jpg图片
            for file in os.listdir(visualize_output_path):  
                if file.endswith('.jpg'):
                    os.remove(os.path.join(visualize_output_path, file))

        for index, frame_data in enumerate(seq_loader):

            # if index % 50 == 0:
            #     print(f"Processing No.{index} image\n")
            
            print(f"Processing No.{index} image\n")
            
            # 获取压缩过后的图片
            img = frame_data['img']
            img_path = frame_data['img_path'][0]

            # 读取cv2的格式
            cv2_img = cv2.imread(img_path)

            # 读取原始图像
            origin_img_path = frame_data['origin_img_path'][0]
            origin_image = Image.open(origin_img_path).convert('RGB')
            origin_image = transforms(origin_image)
            origin_image_width = origin_image.shape[2]
            origin_image_height = origin_image.shape[1]

            # 如果是第一帧，那么根据第一帧的原始图像大小初始化中央凹优化器

            if fovea_width == -1:
                # 获取压缩过后的图像的宽高
                compress_img_width = img[0].shape[2]
                compress_img_height = img[0].shape[1]

                # 根据压缩过后图像宽高计算中央凹区域大小
                # 在选定中央凹区域位置的过程中，
                fovea_width = int(compress_img_width * fovea_scale_width)
                fovea_height = int(compress_img_height * fovea_scale_height)

                print(f"Initialize Fovea. Image Size: {compress_img_width}x{compress_img_height}, Fovea Size: {fovea_width}x{fovea_height}")
                
                # 初始化中央凹优化器
                fovea_optimizer = FoveaOptimizer(img_width=compress_img_width, img_height=compress_img_height, init_image_path=None,
                                                 region_scale=0.025, pixel_change_threshold=70,
                                                 fovea_width=fovea_width, fovea_height=fovea_height,
                                                 is_PIL=True)
            
                

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
                                                                  visualize=False,online_box_type='tlwh')
            
            # 处理边界情况
            if np.isnan(fovea_x) or np.isnan(fovea_y):
                fovea_x = int(origin_img_width / compress_ratio_width / 4)
                fovea_y = int(origin_img_height / compress_ratio_height / 4)
            
            # 将中央凹区域的坐标放大回原始图像的尺寸
            origin_fovea_x = int(fovea_x * compress_ratio_width)
            origin_fovea_y = int(fovea_y * compress_ratio_height)
            origin_fovea_width = int(fovea_width * compress_ratio_width)
            origin_fovea_height = int(fovea_height * compress_ratio_height)

            # 得到在原始图像中，中央凹区域的tlwh
            origin_fovea_pos = (origin_fovea_x, origin_fovea_y, origin_fovea_width, origin_fovea_height)
            # 同时计算在压缩后图像中，中央凹区域的tlwh
            fovea_pos = (fovea_x, fovea_y, fovea_width, fovea_height)

            print(f"Fovea position in origin image in frame No.{index}: {origin_fovea_pos}\n")

            # 截取出给定中央凹区域tlwh对应的原始图像内容
            fovea_img = origin_image[:, origin_fovea_y:origin_fovea_y + origin_fovea_height,
                                    origin_fovea_x:origin_fovea_x + origin_fovea_width]
            # # 用PIL保存fovea_img
            # fovea_img_pil = ToPILImage()(fovea_img)
            # fovea_img_pil.save(f"{visualize_output_root}/{exp_name}/{seq}/{index:06d}_fovea.jpg")
            
            
            _fovea_img = torch.stack([fovea_img], dim=0)

            # 送入中央凹区域的目标检测器进行检测，获得该目标检测器输出的boxes和scores
            _fovea_boxes, _fovea_scores = fovea_obj_detect.detect(_fovea_img)
            # print(f"\tRaw fovea_boxes: {_fovea_boxes}, Raw fovea_scores: {_fovea_scores}")

            fovea_boxes, fovea_scores = [], []

            for i, box in enumerate(_fovea_boxes):
                score = _fovea_scores[i].detach().cpu()
                if score > detect_person_thresh:
                    fovea_boxes.append(box)
                    fovea_scores.append(score)
                # else:
                #     print(f"\t\tFilter out box #{i} {box} with score {score}")
            
            # 将fovea_boxes和fovea_scores转换为Tensor
            fovea_boxes = torch.stack(fovea_boxes, dim=0)
            fovea_scores = torch.stack(fovea_scores, dim=0)
            # print(f"\tfovea_boxes shape: {fovea_boxes.shape}, fovea_scores shape: {fovea_scores.shape}")
            

            # 将所有中央凹区域得到的目标boxes转换到经过压缩的图像的坐标下
            processed_fovea_boxes = get_processed_boxes(fovea_boxes=fovea_boxes, fovea_pos=origin_fovea_pos,
                                                        compress_ratio=[compress_ratio_width, compress_ratio_height])
            processed_fovea_boxes = processed_fovea_boxes.detach().cpu().numpy()

            

            # 对每个processed_fovea_boxes，检查其与所有当前活跃的KCF追踪器的IOU，如果IOU大于0.5，那么认为是同一个目标
            # 不加以处理；否则，新建一个KCF追踪器，并赋予新的id
            
            # 先将所有的active tracker的bbox合并起来
            active_kcf_tracker_bboxes = []
            for kcf_tracker in active_kcf_trackers:
                tracker_bbox = kcf_tracker.current_track['bbox']
                active_kcf_tracker_bboxes.append(tracker_bbox)
            
            # KCF追踪器的bbox是tlwh格式，所以要转换成tlbr格式
            if len(active_kcf_tracker_bboxes) > 0:
                active_kcf_tracker_bboxes = np.asarray(active_kcf_tracker_bboxes)
                active_kcf_tracker_bboxes = tlwh_to_tlbr_numpy(active_kcf_tracker_bboxes)

            # 逐个计算iou

            # 首先保存一份processed_fovea_boxes转换为tlwh的副本
            processed_fovea_boxes_tlwh = tlbr_to_tlwh_numpy(processed_fovea_boxes)
            # 再转换为int
            processed_fovea_boxes_tlwh = processed_fovea_boxes_tlwh.astype(np.int)

            # 在压缩后的图片上可视化中央凹区域和检测到的目标
            cv2_img_clone = cv2_img.copy()
            cv2_fovea_detect_img = visualize_all(cv2_img=cv2_img_clone, fovea_tlwh=fovea_pos, track_tlwhs=processed_fovea_boxes_tlwh,
                                                 track_ids=None, fovea_color=(0, 255, 0))
            
            cv2.imwrite(f"{visualize_output_path}/{index:06d}_detect.jpg", cv2_fovea_detect_img)




            # 如果当前没有任何活跃的kcf追踪器，那么就直接全部新建
            if len(active_kcf_trackers) == 0:
                # print(f"No active KCF trackers, creating new trackers")
                for i, fovea_box in enumerate(processed_fovea_boxes):
                    id_counter += 1
                    new_kcf_tracker = FrogKCFTracker(id_counter)
                    new_bbox = processed_fovea_boxes_tlwh[i]
                    # print(f"New bbox for tracker(No. {id_counter}): {new_bbox}\n")
                    new_kcf_tracker.init_tracker(frame=cv2_img, frame_id=index, bbox=new_bbox)
                    active_kcf_trackers.append(new_kcf_tracker)
                    # print(f"\tNew tracker(No. {id_counter}) created with bbox {new_bbox} at frame No.{index}")
            else:
                for i, fovea_box in enumerate(processed_fovea_boxes):
                    # print(f"Comparing fovea box {fovea_box} with all active KCF trackers")
                    ious = box_iou_numpy(np.expand_dims(fovea_box, axis=0), active_kcf_tracker_bboxes)
                    if (ious < 0.5).all():
                        # 新建一个KCF追踪器
                        id_counter += 1
                        new_kcf_tracker = FrogKCFTracker(id_counter)
                        new_bbox = processed_fovea_boxes_tlwh[i]
                        # 将本帧（压缩后的）内容送入KCF追踪器做初始化
                        new_kcf_tracker.init_tracker(frame=cv2_img, frame_id=index, bbox=new_bbox)
                        active_kcf_trackers.append(new_kcf_tracker)
                        # print(f"\tNew tracker(No. {id_counter}) created with bbox {new_bbox} at frame No.{index}")
            
            # 统计当前帧所有活跃的KCF追踪器的bbox和追踪的id
            active_kcf_tracker_bboxes = []  
            active_kcf_tracker_ids = []
            for kcf_tracker in active_kcf_trackers:
                tracker_bbox = kcf_tracker.current_track['bbox']
                tracker_id = kcf_tracker.id
                active_kcf_tracker_bboxes.append(tracker_bbox)
                active_kcf_tracker_ids.append(tracker_id)
                # print(f"\tActive tracker No. {tracker_id} with bbox {tracker_bbox} at frame No.{index}")
            


            # 绘制出当前帧的追踪结果和中央凹框位置

            # print(f"\tFovea position in compressed image: {fovea_pos}")
            
            cv2_visualize_img = visualize_all(cv2_img=cv2_img, fovea_tlwh=fovea_pos, track_tlwhs=active_kcf_tracker_bboxes,
                                              track_ids=active_kcf_tracker_ids, fovea_color=(0, 255, 0))
            
            # 保存到 visualize_output_root/name/seq_name/ 下，文件名是index前面补0直到文件名长度为6个字符
            
            cv2.imwrite(f"{visualize_output_path}/{index:06d}.jpg", cv2_visualize_img)

            


