"""
Point the released annotation files at your own copy of the videos.

Usage:
  python my_tools/relocate_video_paths.py \
      --anno /path/to/train.json /path/to/test.json \
      --video_dir /path/to/my/videos \
      --out_dir /path/to/my/annotations
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


def index_videos(roots, exts):
    """id (filename without extension) -> sorted list of absolute paths.

    Returns (index, duplicate_ids, empty_files). Paths are sorted so that a run is
    reproducible; which one an entry actually gets is decided in pick_path().
    """
    found = defaultdict(list)
    empty = []
    for root in roots:
        if not os.path.isdir(root):
            sys.exit(f"[relocate] FAILED: not a directory: {root}")
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(path) == 0:
                        empty.append(path)
                        continue
                except OSError:
                    continue
                found[stem].append(path)

    index, dups = {}, []
    for stem, paths in found.items():
        paths.sort()
        index[stem] = paths
        if len(paths) > 1:
            dups.append(stem)
    return index, dups, empty


def pick_path(candidates, original):
    """Choose among several copies of one video id.

    The source dataset keeps some videos in more than one directory, and the copies are
    not byte-identical. When the layout still carries the original directory name, keep
    the entry on the copy its annotation was built from; otherwise take the first path.
    """
    if len(candidates) == 1 or not original:
        return candidates[0]
    original = original[0] if isinstance(original, list) else original
    want = os.path.basename(os.path.dirname(original))
    for path in candidates:
        if os.path.basename(os.path.dirname(path)) == want:
            return path
    return candidates[0]


def entry_id(entry):
    """The video this entry needs. `id` is authoritative; fall back to the shipped path."""
    vid = entry.get("id")
    if vid:
        return str(vid)
    video = entry.get("video")
    video = video[0] if isinstance(video, list) else video
    return os.path.splitext(os.path.basename(video))[0] if video else None


def relocate(anno_path, index, out_dir, keep_missing):
    with open(anno_path) as f:
        data = json.load(f)

    out, missing_ids, dropped_tasks = [], set(), Counter()
    for entry in data:
        vid = entry_id(entry)
        candidates = index.get(vid) if vid else None
        if not candidates:
            missing_ids.add(vid)
            dropped_tasks[entry.get("task", "?")] += 1
            if not keep_missing:
                continue
        else:
            entry["video"] = pick_path(candidates, entry.get("video"))
        out.append(entry)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(anno_path))
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    total_ids = {entry_id(e) for e in data}
    print(f"\n{os.path.basename(anno_path)}")
    print(f"  entries   {len(data)} -> {len(out)}"
          f"{'' if keep_missing else f' ({len(data) - len(out)} dropped)'}")
    print(f"  videos    {len(total_ids) - len(missing_ids)}/{len(total_ids)} found")
    if missing_ids:
        shown = sorted(x for x in missing_ids if x)[:5]
        print(f"  MISSING   {len(missing_ids)} videos, e.g. {', '.join(shown)}")
        print(f"            affected entries by task: {dict(dropped_tasks)}")
        if keep_missing:
            print("            kept with their original paths (--keep_missing); "
                  "training will fail on them")
    print(f"  written   {out_path}")
    return len(missing_ids)


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite the video paths in released annotation files to your own copy.")
    ap.add_argument("--anno", nargs="+", required=True,
                    help="annotation files to convert (e.g. train.json test.json)")
    ap.add_argument("--video_dir", nargs="+", required=True,
                    help="one or more directories holding your videos; searched recursively")
    ap.add_argument("--out_dir", required=True,
                    help="directory for the rewritten files (same basenames)")
    ap.add_argument("--keep_missing", action="store_true",
                    help="keep entries whose video was not found instead of dropping them")
    ap.add_argument("--ext", nargs="+", default=list(VIDEO_EXTS),
                    help=f"video extensions to index (default: {' '.join(VIDEO_EXTS)})")
    args = ap.parse_args()

    for a in args.anno:
        if not os.path.isfile(a):
            sys.exit(f"[relocate] FAILED: annotation not found: {a}")
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}

    print(f"[relocate] indexing videos under: {', '.join(args.video_dir)}")
    index, dups, empty = index_videos(args.video_dir, exts)
    print(f"[relocate] indexed {len(index)} videos")
    if empty:
        print(f"[relocate] WARNING: skipped {len(empty)} zero-byte file(s), "
              f"e.g. {os.path.basename(empty[0])} — re-download these")
    if dups:
        print(f"[relocate] note: {len(dups)} id(s) exist in more than one directory; "
              f"keeping each entry on the copy its annotation names, e.g. {dups[0]}")
    if not index:
        sys.exit("[relocate] FAILED: no videos found — check --video_dir and --ext")

    total_missing = sum(relocate(a, index, args.out_dir, args.keep_missing) for a in args.anno)
    print("\n[relocate] done."
          + ("" if total_missing == 0 else
             "\n[relocate] Some videos were missing. The written files exclude them "
             "(unless --keep_missing), so they are still runnable, but results will "
             "not be comparable with numbers computed on the full set."))


if __name__ == "__main__":
    main()
