import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from ultralytics import YOLO
from numba import jit


def prepare_boxes(boxes, scores, labels, masks):
    """
    Adjust boxes coordinates to be within [0, 1], fix box coordinate ordering,
    and remove boxes (and their corresponding masks) with zero area.
    
    Args:
        boxes (np.ndarray): Array of shape (N, 4) with box coordinates [x1, y1, x2, y2].
        scores (np.ndarray): Array of shape (N,) with scores.
        labels (np.ndarray): Array of shape (N,) with labels.
        masks (list): List of length N containing corresponding masks.
        
    Returns:
        np.ndarray, np.ndarray, np.ndarray, list: Filtered boxes, scores, labels, and masks.
    """
    result_boxes = boxes.copy()

    # Clip coordinates to [0, 1]
    cond = (result_boxes < 0)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Fixed {} boxes coordinates < 0'.format(cond_sum))
        result_boxes[cond] = 0

    cond = (result_boxes > 1)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Fixed {} boxes coordinates > 1. Check that your boxes were normalized at [0, 1]'.format(cond_sum))
        result_boxes[cond] = 1

    # Ensure x1,x2 and y1,y2 are ordered properly
    boxes1 = result_boxes.copy()
    result_boxes[:, 0] = np.min(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(boxes1[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(boxes1[:, [1, 3]], axis=1)

    # Compute area and remove zero-area boxes (and corresponding masks)
    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1])
    cond = (area == 0)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Removed {} boxes with zero area!'.format(cond_sum))
        valid_indices = area > 0
        result_boxes = result_boxes[valid_indices]
        scores = scores[valid_indices]
        labels = labels[valid_indices]
        masks = [masks[i] for i in range(len(masks)) if valid_indices[i]]

    return result_boxes, scores, labels, masks


def cpu_soft_nms_float(dets, sc, Nt, sigma, thresh, method):
    """
    Soft-NMS implementation for boxes with float coordinates in the range [0, 1].

    Args:
        dets (np.ndarray): Array of shape (N, 4) with boxes [x1, y1, x2, y2].
        sc (np.ndarray): Scores for each box.
        Nt (float): IoU threshold for suppression.
        sigma (float): Sigma value for gaussian method.
        thresh (float): Score threshold for keeping boxes.
        method (int): 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS.
        
    Returns:
        np.ndarray: Indices of boxes to keep.
    """
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # Use the coordinates as provided
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = sc.copy()
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0

        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep


@jit(nopython=True)
def nms_float_fast(dets, scores, thresh):
    """
    Fast NMS implementation for boxes with float coordinates in [0, 1].

    Args:
        dets (np.ndarray): Array of shape (N, 4) with boxes [x1, y1, x2, y2].
        scores (np.ndarray): Scores for each box.
        thresh (float): IoU threshold for suppression.
        
    Returns:
        list: Indices of boxes to keep.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_method(boxes, scores, labels, masks, method=3, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """
    Perform NMS or Soft-NMS on detections for each label, and filter corresponding masks.

    Args:
        boxes (list): List of numpy arrays of boxes from each model, each with shape (N_i, 4) in [x1, y1, x2, y2].
        scores (list): List of numpy arrays of scores corresponding to each model's boxes.
        labels (list): List of numpy arrays of labels corresponding to each model's boxes.
        masks (list): List of lists containing masks corresponding to each model's boxes.
        method (int): 1 for linear soft-NMS, 2 for gaussian soft-NMS, 3 for standard NMS.
        iou_thr (float): IoU threshold for suppression.
        sigma (float): Sigma for gaussian soft-NMS.
        thresh (float): Score threshold for keeping boxes.
        weights (list, optional): List of weights for each model.
        
    Returns:
        np.ndarray: Final boxes after NMS.
        np.ndarray: Final scores after NMS.
        np.ndarray: Final labels after NMS.
        list: Final masks corresponding to the kept boxes.
    """
    # If weights are specified, adjust scores
    if weights is not None:
        if len(boxes) != len(weights):
            print('Incorrect number of weights: {}. Must be: {}. Skip it'.format(len(weights), len(boxes)))
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    # Concatenate boxes, scores, and labels from all models
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    
    # Flatten the masks list (each element in masks corresponds to a model's predictions)
    flat_masks = []
    for m in masks:
        flat_masks.extend(m)
    masks = flat_masks

    # Fix coordinates and remove zero-area boxes along with their masks
    boxes, scores, labels, masks = prepare_boxes(boxes, scores, labels, masks)

    # Run NMS independently for each label
    unique_labels = np.unique(labels)
    final_boxes = []
    final_scores = []
    final_labels = []
    final_masks = []
    for l in unique_labels:
        condition = (labels == l)
        boxes_by_label = boxes[condition]
        scores_by_label = scores[condition]
        labels_by_label = np.array([l] * len(boxes_by_label))
        masks_by_label = [m for idx, m in enumerate(masks) if condition[idx]]

        if method != 3:
            keep = cpu_soft_nms_float(boxes_by_label.copy(), scores_by_label.copy(), Nt=iou_thr, sigma=sigma, thresh=thresh, method=method)
        else:
            keep = nms_float_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

        final_boxes.append(boxes_by_label[keep])
        final_scores.append(scores_by_label[keep])
        final_labels.append(labels_by_label[keep])
        final_masks.append([masks_by_label[i] for i in keep])
    
    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)
    final_masks = np.concatenate(final_masks)

    return final_boxes, final_scores, final_labels, final_masks


def nms(boxes, scores, labels, masks, iou_thr=0.5, weights=None):
    """
    Standard NMS wrapper.
    
    Args:
        boxes (list): List of numpy arrays of boxes.
        scores (list): List of numpy arrays of scores.
        labels (list): List of numpy arrays of labels.
        masks (list): List of lists of masks.
        iou_thr (float): IoU threshold for suppression.
        weights (list, optional): List of weights for each model.
        
    Returns:
        np.ndarray: Final boxes.
        np.ndarray: Final scores.
        np.ndarray: Final labels.
        list: Final masks.
    """
    return nms_method(boxes, scores, labels, masks, method=3, iou_thr=iou_thr, weights=weights)


def soft_nms(boxes, scores, labels, masks, method=2, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """
    Soft-NMS wrapper.
     
    Args:
        boxes (list): List of numpy arrays of boxes.
        scores (list): List of numpy arrays of scores.
        labels (list): List of numpy arrays of labels.
        masks (list): List of lists of masks.
        method (int): 1 for linear soft-NMS, 2 for gaussian soft-NMS.
        iou_thr (float): IoU threshold.
        sigma (float): Sigma for gaussian soft-NMS.
        thresh (float): Score threshold.
        weights (list, optional): List of weights for each model.
        
    Returns:
        np.ndarray: Final boxes.
        np.ndarray: Final scores.
        np.ndarray: Final labels.
        list: Final masks.
    """
    return nms_method(boxes, scores, labels, masks, method=method, iou_thr=iou_thr, sigma=sigma, thresh=thresh, weights=weights)

def nms_soft(boxes_list, scores_list, labels_list, masks_list, width, height):
    """
    Apply Soft-NMS to filter overlapping bounding boxes.
    :param boxes_list: List of bounding boxes
    :param scores_list: List of confidence scores
    :param labels_list: List of class labels
    :param width: Frame width
    :param height: Frame height
    :return: Filtered bounding boxes, scores, and labels
    """
    if not boxes_list:
        return [], [], [], []
    
    try:
        filtered_boxes, filtered_scores, filtered_labels, filtered_masks = soft_nms(boxes_list, scores_list, labels_list, masks_list, iou_thr=0.5, sigma=0.5, thresh=0.5, method=2)
        return filtered_boxes, filtered_scores, filtered_labels, filtered_masks
    except ValueError:
        print("Warning: No valid boxes left after Soft-NMS")
        return [], [], [], []

def exponential_smoothing(new_value, prev_value, alpha):
    return alpha * new_value + (1 - alpha) * prev_value

def find_center(mask):
    height, width = mask.shape
    centers = [(int(np.mean(np.where(mask[y, :] > 0)[0])), y) for y in range(height) if np.any(mask[y, :] > 0)]

    return np.array(centers) if len(centers) >= 4 else mask

import numpy as np
import cv2
from scipy.interpolate import splprep, splev

def overlay_bspline_on_mask(centers, mask, bspline_smooth, frame_count, k=3, color=(255, 0, 0), alpha=0.35, smoothness=1500, line_thickness=1):
    if len(mask.shape) == 3:
        height, width, _ = mask.shape
    else:
        height, width = mask.shape
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Fit a cubic B-spline
    tck, u = splprep([centers[:, 0], centers[:, 1]], k=k, s=smoothness)
    u_fine = np.linspace(0, 1, 500)
    x_smooth, y_smooth = splev(u_fine, tck)

    # Apply Exponential Smoothing
    if frame_count == 0 or not isinstance(bspline_smooth, np.ndarray):
        bspline_smooth[:] = np.array(np.column_stack((x_smooth, y_smooth)))  # Ensure it's NumPy

    else:
        bspline_smooth[:, 0] = exponential_smoothing(x_smooth, bspline_smooth[:, 0], alpha)
        bspline_smooth[:, 1] = exponential_smoothing(y_smooth, bspline_smooth[:, 1], alpha)

    bspline_smooth = np.array(bspline_smooth)

    # Convert to integer pixel positions
    x_smooth = np.clip(np.round(bspline_smooth[:, 0]).astype(int), 0, width - 1)
    y_smooth = np.clip(np.round(bspline_smooth[:, 1]).astype(int), 0, height - 1)

    # Overlay B-spline on mask
    for i in range(len(x_smooth) - 1):
        cv2.line(mask, (x_smooth[i], y_smooth[i]), (x_smooth[i + 1], y_smooth[i + 1]), color, line_thickness)
    
    # Find center point along B-spline
    center_y = height // 2
    i_center = np.argmin(np.abs(y_smooth - center_y))
    pt_center = (x_smooth[i_center], y_smooth[i_center])
    
    # Compute angle using first derivative
    dx, dy = splev(u_fine[i_center], tck, der=1)
    angle = np.arctan(dx / dy)
    
    x_offset = pt_center[0] - (width // 2)
    
    # Draw the tangent at the center point in both directions
    L = 20  # Length of the tangent for visualization
    x_tangent_start = int(pt_center[0] - L * dx)
    y_tangent_start = int(pt_center[1] - L * dy)
    x_tangent_end = int(pt_center[0] + L * dx)
    y_tangent_end = int(pt_center[1] + L * dy)
    
    cv2.line(mask, (x_tangent_start, y_tangent_start), (x_tangent_end, y_tangent_end), color, line_thickness)

    return mask, angle, x_offset

def overlay_line_on_mask(points, final_mask, line, frame_count, alpha=0.35, color=(0, 255, 0), thickness=1):
    if len(points) >= 2:
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        m, c = vy / (vx + 1e-6), y - (x * vy / (vx + 1e-6))
        topx, bottomx = int(-c / m), int((final_mask.shape[0] - c) / (m + 1e-6))

        if frame_count == 0:
            line[:] = np.array([topx, bottomx], dtype=np.float32)
        else:
            line[:] = exponential_smoothing(np.array([topx, bottomx], dtype=np.float32), line, alpha)

        if len(final_mask.shape) == 2:
            final_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)

        cv2.line(final_mask, (int(line[0]), 0), (int(line[1]), final_mask.shape[0] - 1), color, thickness)

        angle = np.arctan(1/m)
        x_offset = (line[0] + line[1] - final_mask.shape[1]) / 2

    return final_mask, angle, x_offset

def process_canopy_mask(line, center_line, bspline, canopy_mask, frame_count):
    # Use a large kernel for opening to separate treelines
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, kernel_open)

    # Use a small kernel for dilation to expand treelines
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # closed_mask = cv2.dilate(opened_mask, kernel_dilate, iterations=1)

    # kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # refined_mask = cv2.subtract(closed_mask, cv2.morphologyEx(closed_mask, cv2.MORPH_TOPHAT, kernel_tophat))

    contours, _ = cv2.findContours(opened_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(canopy_mask)
    
    if contours:
        cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

    centers = find_center(final_mask)

    # Overlay B-spline with exponential smoothing
    bspline3_mask, angle_bspline, xoffset_bspline = overlay_bspline_on_mask(centers, final_mask, bspline, frame_count, k=3, color=(0, 0, 255))

    bspline2_mask, angle_line, xoffset_line = overlay_bspline_on_mask(centers, final_mask, bspline, frame_count, k=2, color=(0, 0, 255))#overlay_line_on_mask(points, trajectory_mask, line, frame_count, color=(0, 255, 0))

    
    # Overlay line with exponential smoothing
    points = np.column_stack(np.where(final_mask > 0))[:, ::-1]
    line_mask, angle_center_line, xoffset_center_line = overlay_line_on_mask(points, final_mask, line, frame_count, color=(0, 0, 255))

  
    # Over lay center line with exponential smoothing
    points = centers.reshape(-1, 1, 2).astype(np.float32)
    center_line_mask, angle_center_line, xoffset_center_line = overlay_line_on_mask(points, final_mask, center_line, frame_count, color=(0, 0, 255))

    # print(xoffset_bspline, xoffset_line, xoffset_center_line)
    return bspline3_mask, bspline2_mask, line_mask, center_line_mask, angle_bspline, xoffset_bspline, angle_line, xoffset_line, angle_center_line, xoffset_center_line

