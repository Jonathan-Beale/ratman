"""
build_musepose_dataset.py

Organises single-actor renders into the directory layout expected by
MusePose finetuning:

    <out>/
        dataset/            raw RGB mp4s
        dataset_dwpose/     DWPose skeleton mp4s  (same filenames)
        dataset_dwpose_keypoints/   per-video .npy keypoint files
        meta/
            dataset.json    [{"video_path": ..., "kps_path": ...}, ...]

Each completed camera dir inside renders_dataset/single becomes one
"clip" in the dataset.  The clip name is:
    {model}__{animation}__{camera}   (spaces → underscores)

COCO-17 → OpenPose-18 joint mapping applied during npy conversion.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SRC_DIR    = Path(r"C:\Users\Jon\Desktop\Motion Capture\renders_dataset\single")
OUT_DIR    = Path(r"C:\Users\Jon\Desktop\Motion Capture\musepose_dataset")
DATASET_NAME = "synthetic_single"
# ─────────────────────────────────────────────────────────────────────────────

# COCO-17 index → OpenPose-18 index mapping.
# -1 means the joint must be computed (neck = midpoint of shoulders).
# None means no corresponding joint (left as NaN).
# COCO:  0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear 5=l_sh 6=r_sh
#        7=l_el 8=r_el 9=l_wr 10=r_wr 11=l_hip 12=r_hip
#        13=l_kn 14=r_kn 15=l_an 16=r_an
# OP-18: 0=nose 1=neck 2=r_sh 3=r_el 4=r_wr 5=l_sh 6=l_el 7=l_wr
#        8=r_hip 9=r_kn 10=r_an 11=l_hip 12=l_kn 13=l_an
#        14=r_eye 15=l_eye 16=r_ear 17=l_ear
COCO_TO_OP18 = [
    0,    # OP 0  nose       ← COCO 0
    -1,   # OP 1  neck       ← computed
    6,    # OP 2  r_shoulder ← COCO 6
    8,    # OP 3  r_elbow    ← COCO 8
    10,   # OP 4  r_wrist    ← COCO 10
    5,    # OP 5  l_shoulder ← COCO 5
    7,    # OP 6  l_elbow    ← COCO 7
    9,    # OP 7  l_wrist    ← COCO 9
    12,   # OP 8  r_hip      ← COCO 12
    14,   # OP 9  r_knee     ← COCO 14
    16,   # OP 10 r_ankle    ← COCO 16
    11,   # OP 11 l_hip      ← COCO 11
    13,   # OP 12 l_knee     ← COCO 13
    15,   # OP 13 l_ankle    ← COCO 15
    2,    # OP 14 r_eye      ← COCO 2
    1,    # OP 15 l_eye      ← COCO 1
    4,    # OP 16 r_ear      ← COCO 4
    3,    # OP 17 l_ear      ← COCO 3
]


def convert_keypoints(npz_path: Path, width: int, height: int) -> dict:
    """
    Load our skeleton_coco.npz and return a dict in DWPose / MusePose format:
        {
          'bodies':  {'candidate': (N,18,2) normalised [0,1],
                      'subset':   (N,20)   confidence},
          'hands':   (N,2,21,2)  normalised,
          'faces':   (N,68,2)    normalised  (zeros — we have no face kps),
        }
    All coordinates are normalised to [0, 1] (x/width, y/height).
    NaN joints are kept as NaN; MusePose handles them via the subset mask.
    """
    d = np.load(npz_path, allow_pickle=True)

    body2d   = d["body_kps_2d"].astype(np.float32)   # (N,17,2) pixel
    b_scores = d["body_scores"].astype(np.float32)    # (N,17)
    lh2d     = d["left_hand_kps_2d"].astype(np.float32)   # (N,21,2)
    rh2d     = d["right_hand_kps_2d"].astype(np.float32)  # (N,21,2)

    N = body2d.shape[0]

    # ── Build OpenPose-18 body candidate ─────────────────────────────────────
    op18 = np.full((N, 18, 2), np.nan, dtype=np.float32)
    op18_score = np.zeros((N, 18), dtype=np.float32)

    for op_idx, coco_idx in enumerate(COCO_TO_OP18):
        if coco_idx == -1:
            # Neck: midpoint of left (COCO 5) and right (COCO 6) shoulders
            ls = body2d[:, 5, :]
            rs = body2d[:, 6, :]
            valid = ~(np.isnan(ls[:, 0]) | np.isnan(rs[:, 0]))
            op18[valid, op_idx, :] = (ls[valid] + rs[valid]) * 0.5
            op18_score[valid, op_idx] = (b_scores[valid, 5] + b_scores[valid, 6]) * 0.5
        else:
            src = body2d[:, coco_idx, :]
            op18[:, op_idx, :] = src
            op18_score[:, op_idx] = b_scores[:, coco_idx]

    # Normalise to [0,1]
    op18[:, :, 0] /= width
    op18[:, :, 1] /= height

    # subset: (N, 20) — first 18 are joint confidences, last 2 are unused
    subset = np.full((N, 20), -1.0, dtype=np.float32)
    valid_mask = ~np.isnan(op18[:, :, 0])   # (N,18)
    subset[:, :18] = np.where(valid_mask, op18_score, -1.0)

    # ── Hands: shape (N, 2, 21, 2) normalised ────────────────────────────────
    lh_norm = lh2d.copy()
    rh_norm = rh2d.copy()
    lh_norm[:, :, 0] /= width;  lh_norm[:, :, 1] /= height
    rh_norm[:, :, 0] /= width;  rh_norm[:, :, 1] /= height
    hands = np.stack([lh_norm, rh_norm], axis=1)   # (N,2,21,2)

    # ── Faces: zeros (we don't track face landmarks) ─────────────────────────
    faces = np.zeros((N, 68, 2), dtype=np.float32)

    return {
        "bodies": {"candidate": op18, "subset": subset},
        "hands":  hands,
        "faces":  faces,
    }


def main():
    dataset_dir = OUT_DIR / DATASET_NAME
    dwpose_dir  = OUT_DIR / f"{DATASET_NAME}_dwpose"
    kps_dir     = OUT_DIR / f"{DATASET_NAME}_dwpose_keypoints"
    meta_dir    = OUT_DIR / "meta"

    for d in [dataset_dir, dwpose_dir, kps_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    meta = []
    skipped = done = 0

    # Iterate: single/{model}/{animation}/{scene}/
    for scene_dir in sorted(SRC_DIR.glob("*/*/*")):
        if not scene_dir.is_dir():
            continue

        model_name = scene_dir.parts[-3]
        anim_name  = scene_dir.parts[-2]
        scene_name = scene_dir.parts[-1]

        # Each Camera subdir is one clip
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir():
                continue

            cam_name = cam_dir.name

            # Check required files
            raw_mp4   = scene_dir / f"{cam_name}.mp4"
            dw_mp4    = scene_dir / f"{cam_name}_dwpose.mp4"
            npz_file  = cam_dir   / "skeleton_coco.npz"

            if not (raw_mp4.exists() and dw_mp4.exists() and npz_file.exists()):
                skipped += 1
                continue

            # Safe clip name (no spaces or special chars)
            clip_name = f"{model_name}__{anim_name}__{scene_name}__{cam_name}".replace(" ", "_")

            dst_raw = dataset_dir / f"{clip_name}.mp4"
            dst_dw  = dwpose_dir  / f"{clip_name}.mp4"
            dst_npy = kps_dir     / f"{clip_name}.npy"

            if dst_npy.exists() and dst_raw.exists() and dst_dw.exists():
                # Already done
                meta.append({
                    "video_path": str(dst_raw.relative_to(OUT_DIR)),
                    "kps_path":   str(dst_dw.relative_to(OUT_DIR)),
                })
                done += 1
                continue

            # Copy videos
            shutil.copy2(raw_mp4, dst_raw)
            shutil.copy2(dw_mp4,  dst_dw)

            # Also convert keypoints to npy (for other frameworks)
            frame_pngs = sorted(cam_dir.glob("frame_*.png"))
            if frame_pngs:
                from PIL import Image
                with Image.open(frame_pngs[0]) as im:
                    width, height = im.size
            else:
                width, height = 1024, 1024

            kps_data = convert_keypoints(npz_file, width, height)
            np.save(dst_npy, kps_data)

            meta.append({
                "video_path": str(dst_raw.relative_to(OUT_DIR)),
                "kps_path":   str(dst_dw.relative_to(OUT_DIR)),
            })
            done += 1
            print(f"  [{done}] {clip_name}")

    # Write manifest
    meta_path = meta_dir / f"{DATASET_NAME}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {done} clips written, {skipped} skipped (missing files)")
    print(f"Manifest: {meta_path}  ({len(meta)} entries)")
    print(f"\nTo train, set in your MusePose config:")
    print(f"  meta_paths: [\"{meta_path}\"]")


if __name__ == "__main__":
    main()
