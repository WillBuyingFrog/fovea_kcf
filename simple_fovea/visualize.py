import cv2
import numpy as np

# 创建一个字典，用于存储id和颜色的对应关系
id_color_map = {}

def get_id_color(track_id):
    if track_id not in id_color_map:
        # 如果id不在字典中，则生成一个新的随机颜色，并将其添加到字典中
        id_color_map[track_id] = np.random.uniform(0, 255, size=(3,)).astype(int)
    return (int(id_color_map[track_id][0]), int(id_color_map[track_id][1]), int(id_color_map[track_id][2]))

def visualize_all(cv2_img, fovea_tlwh, track_tlwhs, track_ids, 
                  fovea_color):
    # 在给定的图片上，绘制出中央凹框位置和所有追踪得到的锚框
    # 追踪得到的锚框都有对应的id，每个锚框的颜色和id唯一对应
    # fovea_tlwh: 中央凹框的位置，(top, left, width, height)    
    # track_tlwhs: 追踪得到的锚框的位置，(top, left, width, height)
    # track_ids: 追踪得到的锚框的id
    # fovea_color: 中央凹框的颜色
    # 返回绘制好的图片
    
    # Convert tlwh to tlbr (top left bottom right) for drawing rectangles
    fovea_tlbr = (fovea_tlwh[0], fovea_tlwh[1], fovea_tlwh[0] + fovea_tlwh[2], fovea_tlwh[1] + fovea_tlwh[3])
    track_tlbrs = [(tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]) for tlwh in track_tlwhs]

    # Draw the fovea box
    cv2_img = cv2.rectangle(cv2_img, (int(fovea_tlbr[0]), int(fovea_tlbr[1])), (int(fovea_tlbr[2]), int(fovea_tlbr[3])), fovea_color, 2)
    # print(f"Draw rectangle: {int(fovea_tlbr[1])}, {int(fovea_tlbr[0])}, {int(fovea_tlbr[3])}, {int(fovea_tlbr[2])}")
    

    # 如果track_ids是None，那么只画所有的检测锚框，而不标注id；同时，所有锚框的颜色一样
    if track_ids is None:
        for tlbr in track_tlbrs:
            cv2_img = cv2.rectangle(cv2_img, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), (255, 0, 0), 1)
        return cv2_img


    # Draw the track boxes
    for i, tlbr in enumerate(track_tlbrs):
        id_color = get_id_color(track_ids[i])
        # print(type(id_color[0]))
        # print(f"tlbr: {tlbr}, id_color: {id_color}")
        cv2_img = cv2.rectangle(cv2_img, (tlbr[0], tlbr[1]), (tlbr[2], tlbr[3]), id_color, 1)

        # Add the ID number to the box
        cv2_img = cv2.putText(cv2_img, str(track_ids[i]), (tlbr[0], tlbr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

    return cv2_img