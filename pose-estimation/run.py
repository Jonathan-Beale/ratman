"""
run.py — Unified pose estimation entrypoint.

Selects a backend and runs it on an image or directory of images.

Backends
--------
  hrnet      High-resolution keypoint regression (COCO-17). Requires a
             pre-downloaded checkpoint and the HRNet venv.
  mediapipe  Google MediaPipe Pose. CPU-friendly, no GPU required.
  rtmw       RTMPose whole-body via MMPose + YOLO person detector.
             Supports images and videos.

Usage
-----
  python3 run.py <backend> <input> [options]

Examples
--------
  python3 run.py mediapipe path/to/image.jpg
  python3 run.py mediapipe path/to/images/
  python3 run.py hrnet     path/to/images/  --weights hrnet_weights/pose_hrnet_w32_256x192_official.pth
  python3 run.py rtmw      path/to/video.mp4 --device cpu
  python3 run.py rtmw      path/to/image.jpg --pose-config ./configs/rtmpose.py --pose-weights ./ckpt.pth
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_inputs(path: str) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS | VID_EXTS)
    if p.is_file():
        return [p]
    print(f"Input not found: {path}", file=sys.stderr)
    sys.exit(1)


# ── Backends ──────────────────────────────────────────────────────────────────

def run_mediapipe(inputs: list[Path], output_dir: Path, args):
    import cv2
    try:
        import mediapipe as mp
    except ImportError:
        sys.exit("mediapipe not installed. Run: pip install mediapipe")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose    = mp.solutions.pose
    output_dir.mkdir(parents=True, exist_ok=True)

    with mp_pose.Pose(static_image_mode=True, model_complexity=args.model_complexity) as pose:
        for img_path in inputs:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Skipping unreadable file: {img_path}")
                continue
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                print(f"  No pose detected: {img_path.name}")
                continue
            annotated = img.copy()
            mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out_path = output_dir / f"output_{img_path.name}"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved: {out_path}")


def run_hrnet(inputs: list[Path], output_dir: Path, args):
    hrnet_dir = SCRIPT_DIR / "HRNet"
    sys.path.insert(0, str(hrnet_dir))
    import torch
    import cv2
    import numpy as np
    import yaml
    try:
        from hrnet_model import get_pose_net
    except ImportError:
        sys.exit("hrnet_model not found. Make sure you are running from pose-estimation/ and HRNet/ is present.")

    config_path  = hrnet_dir / "hrnet_config.yaml"
    weights_path = Path(args.weights) if args.weights else hrnet_dir / "hrnet_weights/pose_hrnet_w32_256x192_official.pth"

    if not config_path.exists():
        sys.exit(f"HRNet config not found: {config_path}")
    if not weights_path.exists():
        sys.exit(f"HRNet weights not found: {weights_path}\nPass --weights <path> to specify.")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    connections = [
        (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
    ]

    for img_path in inputs:
        if img_path.suffix.lower() not in IMG_EXTS:
            print(f"  Skipping non-image: {img_path.name}")
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Skipping unreadable: {img_path.name}")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(img_rgb, (192, 256)).astype(np.float32) / 255.0
        inp = (inp - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            heatmaps = model(inp).cpu().numpy()[0]

        kps = []
        for hm in heatmaps:
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            kps.append([idx[1], idx[0], float(hm[idx])])
        kps = np.array(kps)

        sx, sy = w / 48, h / 64
        vis = img_bgr.copy()
        for s, e in connections:
            x1,y1,c1 = kps[s]; x2,y2,c2 = kps[e]
            if c1 > 0.3 and c2 > 0.3:
                cv2.line(vis, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), (0,255,0), 3)
        for x, y, c in kps:
            if c > 0.3:
                cv2.circle(vis, (int(x*sx), int(y*sy)), 6, (0,0,255), -1)

        out_path = output_dir / f"output_{img_path.name}"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path}")


def run_rtmw(inputs: list[Path], output_dir: Path, args):
    rtmw_dir = SCRIPT_DIR / "RTMW"
    sys.path.insert(0, str(rtmw_dir))
    try:
        from MiniPipeline import MiniPipeline, is_image, is_video
    except ImportError:
        sys.exit("Could not import MiniPipeline. Ensure mmpose and ultralytics are installed.")

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = MiniPipeline(
        detector_model=args.detector,
        pose_config=args.pose_config,
        pose_weights=args.pose_weights,
        device="cpu" if args.cpu else args.device,
        yolo_conf=args.yolo_conf,
    )

    for path in inputs:
        print(f"  Processing: {path.name}")
        if is_image(path):
            pipe.process_image(str(path), out_dir=str(output_dir), kpt_thr=args.kpt_thr)
        elif is_video(path):
            pipe.process_video(str(path), out_dir=str(output_dir), kpt_thr=args.kpt_thr)
        else:
            print(f"  Skipping unsupported file type: {path.name}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified pose estimation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("backend", choices=["hrnet", "mediapipe", "rtmw"],
                   help="Pose estimation backend to use")
    p.add_argument("input", help="Path to an image, video, or directory of images")
    p.add_argument("--output-dir", "-o", default="./output",
                   help="Directory to write results (default: ./output)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference")

    # HRNet
    hrnet = p.add_argument_group("HRNet options")
    hrnet.add_argument("--weights", help="Path to HRNet checkpoint .pth file")

    # MediaPipe
    mp_grp = p.add_argument_group("MediaPipe options")
    mp_grp.add_argument("--model-complexity", type=int, default=2, choices=[0,1,2],
                        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy (default: 2)")

    # RTMW
    rtmw = p.add_argument_group("RTMW options")
    rtmw.add_argument("--detector", default="yolov8n.pt",
                      help="YOLO detector weights (default: yolov8n.pt)")
    rtmw.add_argument("--pose-config",
                      default=str(SCRIPT_DIR / "RTMW" / "../mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"),
                      help="MMPose config file for RTMPose model")
    rtmw.add_argument("--pose-weights",
                      default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth",
                      help="MMPose checkpoint path or URL")
    rtmw.add_argument("--device", default="cuda:0",
                      help="Device for MMPose (default: cuda:0)")
    rtmw.add_argument("--yolo-conf", type=float, default=0.25,
                      help="YOLO confidence threshold (default: 0.25)")
    rtmw.add_argument("--kpt-thr", type=float, default=0.3,
                      help="Keypoint draw threshold (default: 0.3)")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    inputs     = collect_inputs(args.input)
    output_dir = Path(args.output_dir)

    print(f"Backend:    {args.backend}")
    print(f"Input:      {args.input}  ({len(inputs)} file(s))")
    print(f"Output dir: {output_dir}")
    print()

    if args.backend == "mediapipe":
        run_mediapipe(inputs, output_dir, args)
    elif args.backend == "hrnet":
        run_hrnet(inputs, output_dir, args)
    elif args.backend == "rtmw":
        run_rtmw(inputs, output_dir, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
