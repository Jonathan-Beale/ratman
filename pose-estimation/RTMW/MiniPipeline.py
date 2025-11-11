"""
MiniPipeline.py (Updated for MMPose v1.x)

Real-Time multi-person whole-body pose estimation pipeline using:
  - Person detection: Ultralytics YOLO (default: yolov8n)
  - Pose estimation: OpenMMLab MMPose with an RTM* whole-body model (e.g., RTMPose-wholebody).

Input:  image (e.g., .jpg, .png, .jpeg) OR video (e.g., .mp4, .mov)
Output: annotated image/video + corresponding .npz with numerical pose data
         - Output_{YYYYmmdd_HHMMSS}.png or .mp4
         - Pose_{YYYYmmdd_HHMMSS}.npz
"""


from __future__ import annotations
import argparse
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


# Ignore, only temporary
# python3 MiniPipeline.py JackBlack.jpg   --pose-config ./configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py   --pose-weights https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth


# YOLO for person detection
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO is required. Install with: pip install ultralytics") from e

# MMPose for whole-body keypoints
try:
    from mmpose.apis import init_model as init_pose_model
    from mmpose.apis import inference_topdown
    from mmpose.structures import PoseDataSample
except Exception as e:
    raise RuntimeError("MMPose is required. Install with: pip install mmpose mmcv") from e

# ----------------------------- Utility helpers -----------------------------
POSE_CONFIG = "../mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"
POSE_WEIGHTS = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS

def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VID_EXTS

# ------------------------------- Drawing -----------------------------------

def draw_bbox_and_keypoints(
    img: np.ndarray,
    bbox_xyxy: np.ndarray,
    keypoints: np.ndarray | None,
    kpt_score_thr: float = 0.3,
) -> None:
    """Draw rectangle and keypoints in-place.
    - bbox_xyxy: [x1,y1,x2,y2]
    - keypoints: (K,3) where last dim includes score
    """
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if keypoints is None:
        return

    K = keypoints.shape[0]
    for i in range(K):
        x, y, s = keypoints[i, 0], keypoints[i, 1], keypoints[i, 2]
        if s >= kpt_score_thr:
            # --- MODIFIED LINE ---
            # Increased radius from 2 to 5 for bigger dots.
            # Changed color from Blue (255,0,0) to bright Yellow (0,255,255).
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)

# ------------------------------- Pipeline ----------------------------------

class MiniPipeline:
    def __init__(
        self,
        detector_model: str = "yolov8n.pt",
        pose_config: str | None = None,
        pose_weights: str | None = None,
        device: str = "cuda:0",
        yolo_conf: float = 0.25,
    ):
        # Load detector
        self.detector = YOLO(detector_model)
        self.yolo_conf = yolo_conf

        # Load pose model (RTMPose-wholebody or similar)
        if pose_config is None or pose_weights is None:
            raise ValueError(
                "You must provide --pose-config and --pose-weights for an RTM* whole-body model from MMPose."
            )
        self.pose_model = init_pose_model(pose_config, pose_weights, device=device)

    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO person detector on a single frame."""
        results = self.detector.predict(source=frame, conf=self.yolo_conf, classes=[0], verbose=False)[0]
        dets = []
        if results.boxes is None or results.boxes.shape[0] == 0:
            return dets
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        for b, s in zip(boxes, scores):
            dets.append({"bbox_xyxy": b.astype(np.float32), "score": float(s)})
        return dets

    def estimate_pose(self, frame: np.ndarray, dets: List[Dict[str, Any]]) -> list[PoseDataSample]:
        """Run MMPose v1.x top-down inference given YOLO detections."""
        if len(dets) == 0:
            return []
        
        # MMPose v1.x API expects a simple numpy array of bounding boxes (N, 4)
        bboxes = np.array([d['bbox_xyxy'] for d in dets], dtype=np.float32)

        # The function call is now updated to use the 'bboxes' keyword argument
        pose_results = inference_topdown(self.pose_model, frame, bboxes=bboxes)
        return pose_results

    # --------------------------- Image processing --------------------------
    def process_image(self, in_path: str, out_dir: str | None = None, kpt_thr: float = 0.3) -> None:
        img = cv2.imread(in_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {in_path}")

        ts = timestamp()
        out_dir = out_dir or str(Path(in_path).parent)
        out_img = Path(out_dir) / f"Output_{ts}.png"
        out_npz = Path(out_dir) / f"Pose_{ts}.npz"

        dets = self.detect_persons(img)
        pose_results = self.estimate_pose(img, dets)

        # Draw and collect data
        frame_boxes = []
        frame_keypoints = []
        frame_scores = []

        for det, pose_result in zip(dets, pose_results):
            bbox_xyxy = det["bbox_xyxy"]
            score = det["score"]

            # Extract keypoints and scores from the PoseDataSample object
            keypoints = pose_result.pred_instances.keypoints[0]
            keypoint_scores = pose_result.pred_instances.keypoint_scores[0]
            kpts = np.hstack([keypoints, keypoint_scores[:, None]])

            draw_bbox_and_keypoints(img, bbox_xyxy, kpts, kpt_thr)

            frame_boxes.append(bbox_xyxy)
            frame_keypoints.append(kpts)
            frame_scores.append(score)

        cv2.imwrite(str(out_img), img)
        np.savez_compressed(
            str(out_npz),
            is_video=False, image_path=in_path,
            boxes=np.array(frame_boxes, dtype=np.float32),
            keypoints=np.array(frame_keypoints, dtype=object),
            scores=np.array(frame_scores, dtype=np.float32),
        )

        print(f"Saved annotated image: {out_img}")
        print(f"Saved pose data:      {out_npz}")

    # --------------------------- Video processing --------------------------
    def process_video(self, in_path: str, out_dir: str | None = None, kpt_thr: float = 0.3) -> None:
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {in_path}")

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ts = timestamp()
        out_dir = out_dir or str(Path(in_path).parent)
        out_vid = Path(out_dir) / f"Output_{ts}.mp4"
        out_npz = Path(out_dir) / f"Pose_{ts}.npz"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (width, height))

        all_boxes, all_keypoints, all_scores = [], [], []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            dets = self.detect_persons(frame)
            pose_results = self.estimate_pose(frame, dets)

            frame_boxes, frame_kpts, frame_scrs = [], [], []

            for det, pose_result in zip(dets, pose_results):
                bbox_xyxy = det["bbox_xyxy"]
                score = det["score"]
                
                # Extract keypoints and scores from the PoseDataSample object
                keypoints = pose_result.pred_instances.keypoints[0]
                keypoint_scores = pose_result.pred_instances.keypoint_scores[0]
                kpts = np.hstack([keypoints, keypoint_scores[:, None]])

                draw_bbox_and_keypoints(frame, bbox_xyxy, kpts, kpt_thr)

                frame_boxes.append(bbox_xyxy)
                frame_kpts.append(kpts)
                frame_scrs.append(score)

            writer.write(frame)

            all_boxes.append(np.array(frame_boxes, dtype=np.float32))
            all_keypoints.append(np.array(frame_kpts, dtype=object))
            all_scores.append(np.array(frame_scrs, dtype=np.float32))

            frame_idx += 1
            if frame_idx % 50 == 0: print(f"Processed {frame_idx} frames…")

        cap.release()
        writer.release()

        np.savez_compressed(
            str(out_npz),
            is_video=True, video_path=in_path, fps=fps,
            boxes=np.array(all_boxes, dtype=object),
            keypoints=np.array(all_keypoints, dtype=object),
            scores=np.array(all_scores, dtype=object),
        )

        print(f"Saved annotated video: {out_vid}")
        print(f"Saved pose data:       {out_npz}")

# --------------------------------- Main ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini pipeline: YOLO person boxes + RTM* whole-body pose via MMPose")
    p.add_argument("input", type=str, help="Path to an input image or video")
    p.add_argument("--output-dir", type=str, default="./Outputs/", help="Optional output directory (defaults to input's folder)")
    p.add_argument("--detector", type=str, default="yolov8n.pt", help="Ultralytics YOLO weights (for person detection)")
    p.add_argument("--pose-config", type=str, default=POSE_CONFIG, help="MMPose config for RTM* whole-body model")
    p.add_argument("--pose-weights", type=str, default=POSE_WEIGHTS, help="Checkpoint path/URL for the pose model")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for MMPose model, e.g. 'cuda:0' or 'cpu'")
    p.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold for person detection")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint draw threshold (min score to render)")
    return p.parse_args()


def main():
    args = parse_args()

    in_path = args.input
    if not Path(in_path).exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    pipe = MiniPipeline(
        detector_model=args.detector,
        pose_config=args.pose_config,
        pose_weights=args.pose_weights,
        device=args.device,
        yolo_conf=args.yolo_conf,
    )

    if is_image(in_path):
        pipe.process_image(in_path, out_dir=args.output_dir, kpt_thr=args.kpt_thr)
    elif is_video(in_path):
        pipe.process_video(in_path, out_dir=args.output_dir, kpt_thr=args.kpt_thr)
    else:
        print("Unsupported input type. Provide an image or a video file.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()