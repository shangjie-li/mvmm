import torch

from ops.iou3d_nms.iou3d_nms_utils import nms_gpu


def nms(scores, boxes, score_thresh, nms_thresh):
    pre_max_size = 4096
    post_max_size = 500
    
    src_scores = scores
    mask = scores >= score_thresh
    scores = scores[mask]
    boxes = boxes[mask]

    selected = []
    if scores.shape[0] > 0:
        scores_for_nms, indices = torch.topk(scores, k=min(pre_max_size, scores.shape[0]))
        boxes_for_nms = boxes[indices]
        keep_indices, selected_scores = nms_gpu(boxes_for_nms[:, 0:7], scores_for_nms, nms_thresh)
        selected = indices[keep_indices[:post_max_size]]

    original_indices = mask.nonzero().view(-1)
    selected = original_indices[selected]

    return selected, src_scores[selected]
