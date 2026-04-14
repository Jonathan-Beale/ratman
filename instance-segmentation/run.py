"""
run.py — Unified instance segmentation entrypoint.

Selects a backend and runs it on an image, video, or directory.

Backends
--------
  maskrcnn   Mask R-CNN (torchvision). Image input only.
             Requires a pre-trained .pth checkpoint.
  yolov8     YOLOv8 segmentation via Ultralytics. Image and video.
             Launches a Gradio web UI by default; use --no-ui for CLI mode.
  yolov12    YOLOv12 segmentation via Ultralytics. Image and video.
             Designed for person segmentation.

Usage
-----
  python3 run.py <backend> <input> [options]

Examples
--------
  python3 run.py maskrcnn  path/to/images/ --weights Mask_RCNN/maskrcnn_coco_model.pth
  python3 run.py yolov8    path/to/image.jpg
  python3 run.py yolov8    path/to/video.mp4 --model yolov8n-seg.pt
  python3 run.py yolov8    --ui                          # launch Gradio web UI
  python3 run.py yolov12   path/to/video.mp4
  python3 run.py yolov12   path/to/images/ --model yolo12l-person-seg.pt
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

def run_maskrcnn(inputs: list[Path], output_dir: Path, args):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torchvision.transforms as T

    sys.path.insert(0, str(SCRIPT_DIR / "Mask_RCNN"))
    try:
        from inference_script import get_model, COCO_CLASSES
    except ImportError:
        sys.exit("Could not import Mask_RCNN/inference_script.py. Ensure torchvision is installed.")

    weights = Path(args.weights) if args.weights else SCRIPT_DIR / "Mask_RCNN/maskrcnn_coco_model.pth"
    if not weights.exists():
        sys.exit(f"Mask R-CNN weights not found: {weights}\nPass --weights <path> to specify.")

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"  Device: {device}")

    model = get_model(num_classes=91)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = T.ToTensor()

    for img_path in inputs:
        if img_path.suffix.lower() not in IMG_EXTS:
            print(f"  Skipping non-image: {img_path.name}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).to(device)
        import torch
        with torch.no_grad():
            preds = model([img_tensor])[0]

        keep   = preds["scores"] > args.conf
        boxes  = preds["boxes"][keep].cpu().numpy()
        labels = preds["labels"][keep].cpu().numpy()
        scores = preds["scores"][keep].cpu().numpy()
        masks  = preds["masks"][keep].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)
        colors = plt.cm.hsv(np.linspace(0, 1, max(len(boxes), 1))).tolist()

        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=colors[i], linewidth=2)
            ax.add_patch(rect)
            m = mask[0] > 0.5
            colored = np.zeros((*m.shape, 4))
            colored[m] = [*colors[i][:3], 0.5]
            ax.imshow(colored)
            cls = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"cls{label}"
            ax.text(x1, y1-5, f"{cls}: {score:.2f}", color="white", fontsize=9,
                    bbox=dict(facecolor=colors[i], alpha=0.7, edgecolor="none", pad=1))

        ax.axis("off")
        plt.tight_layout()
        out_path = output_dir / f"output_{img_path.stem}.png"
        plt.savefig(str(out_path), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {out_path}  ({len(boxes)} instances)")


def run_yolov8(inputs: list[Path], output_dir: Path, args):
    if args.ui:
        # Launch Gradio web UI from the existing app.py
        sys.path.insert(0, str(SCRIPT_DIR / "yolov8"))
        try:
            from app import gradio_app
        except ImportError:
            sys.exit("Could not import yolov8/app.py. Ensure gradio and ultralytics are installed.")
        gradio_app.launch(share=False)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")
    import cv2

    model = YOLO(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        ext = path.suffix.lower()
        if ext in IMG_EXTS:
            results = model.predict(source=str(path), imgsz=args.imgsz, conf=args.conf,
                                    device="cpu" if args.cpu else args.device)
            out_path = output_dir / f"output_{path.name}"
            annotated = results[0].plot()
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved: {out_path}")

        elif ext in VID_EXTS:
            cap = cv2.VideoCapture(str(path))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = output_dir / f"output_{path.stem}.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf,
                                        device="cpu" if args.cpu else args.device, verbose=False)
                writer.write(results[0].plot())
            cap.release()
            writer.release()
            print(f"  Saved: {out_path}")
        else:
            print(f"  Skipping unsupported file: {path.name}")


def run_yolov12(inputs: list[Path], output_dir: Path, args):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")
    import cv2

    model = YOLO(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        ext = path.suffix.lower()
        if ext in IMG_EXTS:
            results = model.predict(source=str(path), imgsz=args.imgsz, conf=args.conf,
                                    device="cpu" if args.cpu else args.device, verbose=False)
            out_path = output_dir / f"output_{path.name}"
            cv2.imwrite(str(out_path), results[0].plot())
            print(f"  Saved: {out_path}")

        elif ext in VID_EXTS:
            cap    = cv2.VideoCapture(str(path))
            fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = output_dir / f"output_{path.stem}.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, verbose=False)
                writer.write(results[0].plot())
            cap.release()
            writer.release()
            print(f"  Saved: {out_path}")
        else:
            print(f"  Skipping unsupported file: {path.name}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified instance segmentation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("backend", choices=["maskrcnn", "yolov8", "yolov12"],
                   help="Segmentation backend to use")
    p.add_argument("input", nargs="?",
                   help="Path to an image, video, or directory (not required with --ui)")
    p.add_argument("--output-dir", "-o", default="./output",
                   help="Directory to write results (default: ./output)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold (default: 0.5)")

    # Mask R-CNN
    mrcnn = p.add_argument_group("Mask R-CNN options")
    mrcnn.add_argument("--weights",
                       help="Path to Mask R-CNN .pth checkpoint")

    # YOLO shared
    yolo = p.add_argument_group("YOLO options (yolov8 / yolov12)")
    yolo.add_argument("--model", default=None,
                      help="YOLO weights file (default: yolov8n-seg.pt for yolov8, yolo12l-person-seg.pt for yolov12)")
    yolo.add_argument("--imgsz", type=int, default=640,
                      help="Inference image size (default: 640)")
    yolo.add_argument("--device", default="0",
                      help="CUDA device index for YOLO (default: 0)")

    # YOLOv8 UI
    yolo8 = p.add_argument_group("YOLOv8 options")
    yolo8.add_argument("--ui", action="store_true",
                       help="Launch Gradio web UI instead of CLI inference")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Set default YOLO model per backend
    if args.model is None:
        args.model = "yolov8n-seg.pt" if args.backend == "yolov8" else "yolo12l-person-seg.pt"

    # --ui doesn't need an input path
    if not args.input and not (args.backend == "yolov8" and args.ui):
        parser.error("the following argument is required: input")

    output_dir = Path(args.output_dir)

    if args.input:
        inputs = collect_inputs(args.input)
        print(f"Backend:    {args.backend}")
        print(f"Input:      {args.input}  ({len(inputs)} file(s))")
        print(f"Output dir: {output_dir}")
        print()
    else:
        inputs = []

    if args.backend == "maskrcnn":
        run_maskrcnn(inputs, output_dir, args)
    elif args.backend == "yolov8":
        run_yolov8(inputs, output_dir, args)
    elif args.backend == "yolov12":
        run_yolov12(inputs, output_dir, args)

    if inputs:
        print("\nDone.")


if __name__ == "__main__":
    main()
