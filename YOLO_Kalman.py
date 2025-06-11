#!/bin/python3
import cv2
from ultralytics import YOLO
from Post_Processing import process_canopy_mask, nms_soft, exponential_smoothing
import numpy as np
import time


model = YOLO("canopy11s640.engine", task="segment")

kalman = cv2.KalmanFilter(4, 4)
kalman.measurementMatrix = np.eye(4, dtype=np.float32)
kalman.transitionMatrix = np.eye(4, dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

IMG_WD = 640
IMG_HT = 384


bspline = []
tree_line = np.zeros(2, dtype=np.float32)
tree_center_line = np.zeros(2, dtype=np.float32) 

def control_drone(mask, angle, xoffset, prev_yaw, prev_roll, alpha_yaw=0.3, alpha_roll=0.3, cam_FOV=90):
    yaw = angle * 180 / np.pi
    smoothed_yaw = exponential_smoothing(yaw, prev_yaw, alpha_yaw)

    roll = (-xoffset) * cam_FOV / mask.shape[1]
    
    smoothed_roll = exponential_smoothing(roll, prev_roll, alpha_roll)

    return float(smoothed_yaw), float(smoothed_roll)

def print_and_select_bbox(detections):
    centers = {i+1: (int((d[0]+d[2])/2), int((d[1]+d[3])/2)) for i, d in enumerate(detections)}
    for idx, center in centers.items():
        print(f"{idx} - {center}")
    selected_id = int(input("Select bbox ID: "))
    return selected_id

def kalman_filter_tracking(selected_bbox, selected_mask, im0, detections, masks):
    distances = [np.linalg.norm(np.array(selected_bbox[:2]) - np.array(d[:2])) for d in detections]
    selected_bbox = detections[np.argmin(distances)]
    selected_mask = masks[np.argmin(distances)]

    x1, y1, x2, y2 = map(int, selected_bbox)
    measured = np.array([[x1], [y1], [x2], [y2]], dtype=np.float32)

    predicted = kalman.predict()
    kalman.correct(measured * 0.15 + predicted * 0.85)

    px1, py1, px2, py2 = map(int, predicted)
    cv2.rectangle(im0, (px1, py1), (px2, py2), (0, 255, 255), 2)
    cv2.putText(im0, "Kalman", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.putText(im0, "Detection", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return selected_mask

def run_yolo(cap):


    selected_bbox = None
    printed_once = False
    frame_count = 0
    # Initialize previous yaw and roll values
    prev_yaw_st, prev_roll_st = 0, 0
    prev_yaw_spline, prev_roll_spline = 0, 0

    while True:
        start_time = time.time()

        ret, im0 = cap.read()
        if not ret:
            break

        # Resize frame to the model's expected input
        #im01 = cv2.rotate(im0, cv2.ROTATE_90_CLOCKWISE, im0)
        im0 = cv2.resize(im0, (640, 384))
        #cv2.imshow("Original", im01)

        yolo_start = time.time()
        results = model.predict(
            im0,
            iou=.95,
            conf=0.35,
            classes=[0],
            stream=False,
            device="cuda:0",
            verbose=False,
            half=True
        )

        #res = model.predict(
        #    im0.copy(),
        #    iou=.25,
        #    conf=.25,
        #    classes=[0],
        #    stream=False,
        #    device="cuda:0",
        #    verbose=False,
        #    half=True
        #)

        yolo_time = time.time() - yolo_start
        
        postprocess_start = time.time()
        # Initialize lists to store detections
        boxes_list, scores_list, labels_list, masks_list = [], [], [], []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue  # No detections in this result

            # Extract detection information as NumPy arrays
            boxes = result.boxes.xyxy.cpu().numpy()  # shape: (N, 4)
            scores = result.boxes.conf.cpu().numpy()   # shape: (N,)
            labels = result.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)
            masks = result.masks.data.cpu().numpy()    # shape: (N, mask_H, mask_W)

            # Normalize bounding boxes to [0, 1]
            boxes[:, [0, 2]] /= IMG_WD  # Normalize x_min and x_max
            boxes[:, [1, 3]] /= IMG_HT  # Normalize y_min and y_max

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
            masks_list.append(masks)

        num_detections = sum(b.shape[0] for b in boxes_list)
        
        if num_detections > 0:
            # Use your NMS/Soft-NMS function that now takes masks as input.
            # Note: Ensure that your `nms_soft` function expects NumPy arrays rather than lists.
            filtered_boxes, filtered_scores, filtered_labels, filtered_masks = nms_soft(
                boxes_list, scores_list, labels_list, masks_list, IMG_WD, IMG_HT
            )

            # Draw detections on the frame
            if len(filtered_boxes):
                for box, score, label, mask in zip(filtered_boxes, filtered_scores, filtered_labels, filtered_masks):
                    # Convert normalized coordinates back to pixel values
                    x_min, y_min, x_max, y_max = (np.array(box) * [IMG_WD, IMG_HT, IMG_WD, IMG_HT]).astype(int)

                    cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    text = f"{label}: {score:.2f}"
                    cv2.putText(im0, text, (x_min, max(y_min - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"Detections -> NMS_Soft Detections: {num_detections}, {len(filtered_boxes)}")

        # Display results from the model's built-in plot (if available)
        annotated_frame = results[0].plot()  # Assuming the first result contains a plot
        #annotated_fr = res[0].plot()  # Assuming the first result contains a plot

        # time.sleep(0.25)
        #cv2.imshow("YOLOv11 - Initial Detections", annotated_frame)
        #cv2.imshow("Hard", annotated_fr)
        #cv2.imshow("YOLOv11 + Kalman Filter", im0)

        canopy_mask = np.zeros((384, 640), dtype=np.uint8)
        
        if num_detections > 0:
            detections = (np.array(filtered_boxes) * [IMG_WD, IMG_HT, IMG_WD, IMG_HT]).astype(int)

            if not printed_once:
                print("Displaying initial detections... Press any key to proceed with bbox selection.")
                cv2.waitKey(0)
                selected_id = print_and_select_bbox(detections)
                selected_bbox = detections[selected_id - 1]
                selected_mask = filtered_masks[selected_id - 1]
                printed_once = True

            # Update the selected mask using tracking (your kalman_filter_tracking implementation)
            selected_mask = kalman_filter_tracking(selected_bbox, selected_mask, im0, detections, filtered_masks)

            canopy_mask[selected_mask > 0] = 1
            canopy_mask = cv2.resize(canopy_mask, (IMG_WD, IMG_HT), interpolation=cv2.INTER_AREA)
            bspline3_mask, bspline2_mask, line_mask, center_line_mask, angle_bspline, xoffset_bspline, angle_line, xoffset_line, angle_center_line, xoffset_center_line = process_canopy_mask(tree_line, tree_center_line, bspline, canopy_mask, frame_count)
            yaw_st, roll_st = control_drone(canopy_mask, angle_center_line, xoffset_center_line, prev_yaw_st, prev_roll_st)
            yaw_spline, roll_spline = control_drone(canopy_mask, angle_bspline, xoffset_bspline, prev_yaw_spline, prev_roll_spline)
        
        else:
            yaw_st, roll_st = prev_yaw_st, prev_roll_st
            yaw_spline, roll_spline = prev_yaw_spline, prev_roll_spline

        prev_yaw_st, prev_roll_st = yaw_st, roll_st
        prev_yaw_spline, prev_roll_spline = yaw_spline, roll_spline

        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - start_time
        fps = 1 / total_time


        print(f"YOLOv11 Inference Time: {yolo_time:.4f}s, FPS: {1/yolo_time:.2f}")
        print(f"Post-Processing Time: {postprocess_time:.4f}s, FPS: {1/postprocess_time:.2f}")
        print(f"Total Processing Time: {total_time:.4f}s, FPS: {1/total_time:.2f}")

        cv2.putText(im0, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, f"Yaw: {yaw_st:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, f"Roll: {roll_st:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, f"Yaw: {yaw_spline:.2f}", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, f"Roll: {roll_spline:.2f}", (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(bspline3_mask, (IMG_WD//2, IMG_HT//2), 2, (255, 0, 0), -1)
        cv2.circle(bspline2_mask, (IMG_WD//2, IMG_HT//2), 2, (255, 0, 0), -1)
        cv2.circle(line_mask, (IMG_WD//2, IMG_HT//2), 2, (255, 0, 0), -1)
        cv2.circle(center_line_mask, (IMG_WD//2, IMG_HT//2), 2, (255, 0, 0), -1)

        
        
        cv2.imshow("Canopy Tracking", bspline2_mask)
        cv2.imshow("YOLOv11 + Kalman Filter", im0)

        
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            # cv2.imwrite(f"hard1.png", annotated_fr)
            # cv2.imwrite(f"overall1.png", annotated_frame)
            # cv2.imwrite(f"soft1.png", im0)

            
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cv2.VideoCapture("DJI_0008.MOV")
    run_yolo(cap)