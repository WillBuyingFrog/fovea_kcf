


def tlbr_to_tlwh(bboxes):

    # 输入形状：[n, 4]，pytorch tensor
    # 输出形状：[n, 4]， pytorch tensor
    ret = bboxes.clone()
    ret[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    ret[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return ret

def tlwh_to_tlbr(bboxes):
    # 输入形状：[n, 4], pytorch tensor
    # 输出形状：[n, 4], pytorch tensor
    ret = bboxes.clone()
    ret[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    ret[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return ret


def tlbr_to_tlwh_numpy(bboxes):
    # 输入形状：[n, 4], numpy array
    # 输出形状：[n, 4], numpy array
    ret = bboxes.copy()
    ret[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    ret[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return ret

def tlwh_to_tlbr_numpy(bboxes):
    # 输入形状：[n, 4], numpy array
    # 输出形状：[n, 4], numpy array
    ret = bboxes.copy()
    ret[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    ret[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return ret