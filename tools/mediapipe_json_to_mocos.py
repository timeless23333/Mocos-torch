import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


MP = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_index": 19,
    "right_index": 20,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def point(landmarks, name):
    lm = landmarks[MP[name]]
    return np.array([lm["x"], lm["y"], lm["z"]], dtype=np.float32)


def midpoint(a, b):
    return (a + b) / 2.0


def mediapipe_to_kinect20(landmarks):
    left_hip = point(landmarks, "left_hip")
    right_hip = point(landmarks, "right_hip")
    left_shoulder = point(landmarks, "left_shoulder")
    right_shoulder = point(landmarks, "right_shoulder")

    hip_center = midpoint(left_hip, right_hip)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    spine = midpoint(hip_center, shoulder_center)

    left_ear = point(landmarks, "left_ear")
    right_ear = point(landmarks, "right_ear")
    head = midpoint(left_ear, right_ear)
    if not np.isfinite(head).all():
        head = point(landmarks, "nose")

    joints = np.stack(
        [
            hip_center,
            spine,
            shoulder_center,
            head,
            left_shoulder,
            point(landmarks, "left_elbow"),
            point(landmarks, "left_wrist"),
            point(landmarks, "left_index"),
            right_shoulder,
            point(landmarks, "right_elbow"),
            point(landmarks, "right_wrist"),
            point(landmarks, "right_index"),
            left_hip,
            point(landmarks, "left_knee"),
            point(landmarks, "left_ankle"),
            point(landmarks, "left_foot_index"),
            right_hip,
            point(landmarks, "right_knee"),
            point(landmarks, "right_ankle"),
            point(landmarks, "right_foot_index"),
        ],
        axis=0,
    )
    return joints


def parse_person_id(path):
    match = re.search(r"p(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"p(\d+)", path.parent.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse person id from {path}")
    return int(match.group(1))


def parse_view(path):
    name = path.stem.lower()
    for view in ("left", "right", "front", "back"):
        if view in name:
            return view
    raise ValueError(f"Cannot parse view from {path}")


def parse_take(path):
    match = re.search(r"_(\d+)$", path.stem)
    if not match:
        match = re.search(r"take[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def parse_csv_set(value, caster=str):
    if value is None:
        return None
    parsed = {caster(v.strip()) for v in value.split(",") if v.strip()}
    return parsed or None


def load_sequence(path, landmark_key, min_visibility):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    for frame in data.get("frames", []):
        poses = frame.get("poses") or []
        if not poses:
            continue
        pose = poses[0]
        landmarks = pose.get(landmark_key)
        if not landmarks or len(landmarks) < 33:
            continue
        visibility = [
            landmarks[i].get("visibility", 1.0)
            for i in (
                MP["left_shoulder"],
                MP["right_shoulder"],
                MP["left_hip"],
                MP["right_hip"],
                MP["left_knee"],
                MP["right_knee"],
                MP["left_ankle"],
                MP["right_ankle"],
            )
        ]
        if min(visibility) < min_visibility:
            continue
        frames.append(mediapipe_to_kinect20(landmarks))
    if not frames:
        return np.empty((0, 20, 3), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32)


def clips_from_sequences(items, length, stride):
    clips = []
    labels = []
    ids = {}
    for person_id, sequence in items:
        usable = 0
        for start in range(0, max(sequence.shape[0] - length + 1, 0), stride):
            clip = sequence[start : start + length]
            clip_index = len(clips)
            clips.append(clip)
            labels.append(person_id)
            ids.setdefault(person_id, []).append(clip_index)
            usable += 1
        if usable == 0:
            print(f"warning: person {person_id} sequence has only {sequence.shape[0]} valid frames; skipped")

    if not clips:
        raise RuntimeError("No clips generated. Lower --length/--stride or check JSON landmarks.")
    return np.stack(clips, axis=0), np.array(labels, dtype=np.int64), ids


def save_split(out_dir, dataset_name, length, clips, labels, ids, test):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "t_source" if test else "source"
    flat = clips.reshape(-1, 20, 3)
    np.save(out_dir / f"{prefix}_x_{dataset_name}_{length}.npy", flat[:, :, 0])
    np.save(out_dir / f"{prefix}_y_{dataset_name}_{length}.npy", flat[:, :, 1])
    np.save(out_dir / f"{prefix}_z_{dataset_name}_{length}.npy", flat[:, :, 2])
    np.save(out_dir / f"ids_{dataset_name}_{length}.npy", ids)
    np.save(out_dir / f"frame_id_{dataset_name}_{length}.npy", labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="Datasets/raw_json")
    parser.add_argument("--output-root", default="Datasets/MYDATA/6")
    parser.add_argument("--dataset-name", default="MYDATA")
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--gallery-view", default="left")
    parser.add_argument("--probe-views", default="right,front,back")
    parser.add_argument("--train-views", default=None)
    parser.add_argument("--train-takes", default=None)
    parser.add_argument("--gallery-takes", default=None)
    parser.add_argument("--probe-takes", default=None)
    parser.add_argument("--gallery-split", default="Still")
    parser.add_argument("--probe-split", default="Walking")
    parser.add_argument("--landmark-key", default="landmarks_3d_world")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} exists; pass --force to replace it")
        shutil.rmtree(output_root)

    probe_views = parse_csv_set(args.probe_views.lower())
    train_views = parse_csv_set(args.train_views.lower()) if args.train_views else None
    train_takes = parse_csv_set(args.train_takes, int)
    gallery_takes = parse_csv_set(args.gallery_takes, int)
    probe_takes = parse_csv_set(args.probe_takes, int)
    gallery_items = []
    probe_items = []
    train_items = []

    for path in sorted(input_root.rglob("*.json")):
        person_id = parse_person_id(path)
        view = parse_view(path)
        take = parse_take(path)
        seq = load_sequence(path, args.landmark_key, args.min_visibility)
        print(f"{path}: person={person_id} view={view} take={take} valid_frames={seq.shape[0]}")

        train_view_match = view in train_views if train_views is not None else view == args.gallery_view.lower()
        if train_view_match and (train_takes is None or take in train_takes):
            train_items.append((person_id, seq))

        if view == args.gallery_view.lower() and (gallery_takes is None or take in gallery_takes):
            gallery_items.append((person_id, seq))

        if view in probe_views and (probe_takes is None or take in probe_takes):
            probe_items.append((person_id, seq))

    train_clips, train_labels, train_ids = clips_from_sequences(train_items, args.length, args.stride)
    gallery_clips, gallery_labels, gallery_ids = clips_from_sequences(gallery_items, args.length, args.stride)
    probe_clips, probe_labels, probe_ids = clips_from_sequences(probe_items, args.length, args.stride)

    train_dir = output_root / "train_npy_data"
    gallery_dir = output_root / "test_npy_data" / args.gallery_split
    probe_dir = output_root / "test_npy_data" / args.probe_split

    save_split(train_dir, args.dataset_name, args.length, train_clips, train_labels, train_ids, test=False)
    save_split(gallery_dir, args.dataset_name, args.length, gallery_clips, gallery_labels, gallery_ids, test=True)
    save_split(probe_dir, args.dataset_name, args.length, probe_clips, probe_labels, probe_ids, test=True)

    print("written:")
    print(f"  train clips: {train_clips.shape[0]}")
    print(f"  gallery clips: {gallery_clips.shape[0]}")
    print(f"  probe clips: {probe_clips.shape[0]}")
    print(f"  output: {output_root}")


if __name__ == "__main__":
    main()
