# build_lmdb.py
import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


SUFFIXES = ["A", "D", "S", "T"]

def imread_bytes(path: str) -> bytes:
    # 用 cv2 读，然后重新编码成 png 存 bytes，避免原始文件格式/压缩差异导致读取不一致
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    # 统一存 png（无损，支持 1/3/4 通道，uint8/uint16）
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    return buf.tobytes()

def decode_png_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="SHIQ train dir")
    ap.add_argument("--out", required=True, help="output lmdb dir")
    ap.add_argument("--ext", default="", help="optional: only include files with this ext (e.g. .png)")
    ap.add_argument("--map_size_gb", type=float, default=80.0, help="lmdb map size in GB (adjust if needed)")
    ap.add_argument("--dry_run", action="store_true", help="only scan and report, do not write lmdb")
    args = ap.parse_args()

    src = Path(args.src)
    assert src.exists(), f"src not found: {src}"

    # 递归找文件
    ex = args.ext.lower().strip()
    all_files = [p for p in src.rglob("*") if p.is_file()]
    if ex:
        all_files = [p for p in all_files if p.suffix.lower() == ex]

    print(f"[SCAN] src={src}")
    print(f"[SCAN] total files found: {len(all_files)} (ext filter={ex or 'none'})")

    # 解析：从文件名中识别 (object_id, suffix)
    # 支持：xxx_A.png 或 xxx-A.jpg 或 xxx_A_XXX.png（尽量鲁棒）
    # 以最后一个 _A/_D/_S/_T 为准
    pattern = re.compile(r"^(.*?)[_\-]([ADST])$", re.IGNORECASE)

    groups = defaultdict(dict)  # obj_id -> {suffix: path}
    bad = 0

    for p in all_files:
        stem = p.stem  # no extension
        m = pattern.match(stem)
        if not m:
            continue
        obj_id = m.group(1)
        suf = m.group(2).upper()
        if suf not in SUFFIXES:
            continue
        # 同一个 obj_id+suf 如果重复，记录
        if suf in groups[obj_id]:
            print(f"[WARN] duplicate for {obj_id}_{suf}: {groups[obj_id][suf]} vs {p}")
        groups[obj_id][suf] = str(p)

    obj_ids = sorted(groups.keys())
    print(f"[GROUP] matched object ids: {len(obj_ids)}")

    # 检查完整性
    complete = []
    incomplete = []
    for oid in obj_ids:
        missing = [s for s in SUFFIXES if s not in groups[oid]]
        if missing:
            incomplete.append((oid, missing))
        else:
            complete.append(oid)

    print(f"[CHECK] complete objects: {len(complete)}")
    print(f"[CHECK] incomplete objects: {len(incomplete)}")
    if incomplete[:10]:
        print("[CHECK] examples of incomplete:")
        for oid, missing in incomplete[:10]:
            print(f"  - {oid}: missing {missing}")

    if args.dry_run:
        print("[DRY_RUN] stop here.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    map_size = int(args.map_size_gb * (1024**3))
    env = lmdb.open(str(out), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False)

    meta = {
        "src": str(src),
        "suffixes": SUFFIXES,
        "num_complete": len(complete),
        "keys": [],
    }

    print(f"[LMDB] writing to: {out}")
    with env.begin(write=True) as txn:
        # 写 meta 占位，后面再覆盖
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    # 分批写，避免事务太大
    BATCH = 256
    wrote = 0

    for i in tqdm(range(0, len(complete), BATCH), desc="write lmdb"):
        batch = complete[i:i+BATCH]
        with env.begin(write=True) as txn:
            for oid in batch:
                try:
                    paths = groups[oid]
                    # 四张图分别存：f"{oid}/{suf}"
                    for suf in SUFFIXES:
                        k = f"{oid}/{suf}".encode("utf-8")
                        v = imread_bytes(paths[suf])
                        txn.put(k, v)
                    # 记录 key（object id）
                    meta["keys"].append(oid)
                    wrote += 1
                except Exception as e:
                    bad += 1
                    print(f"[ERROR] oid={oid} build failed: {e}")

        # 让你知道进度/坏样本数
        if wrote % 1000 == 0 and wrote > 0:
            print(f"[PROGRESS] wrote={wrote}, bad={bad}")

    # 写最终 meta
    with env.begin(write=True) as txn:
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    env.sync()
    env.close()

    # 同时写出一个可读的 meta.json
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote objects={wrote}, bad={bad}")
    print(f"[DONE] meta.json at: {out/'meta.json'}")

    # 额外：快速抽样验证一次 decode
    if meta["keys"]:
        test_oid = meta["keys"][0]
        env = lmdb.open(str(out), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            b = txn.get(f"{test_oid}/A".encode("utf-8"))
            img = decode_png_bytes(b)
            print(f"[VERIFY] first key={test_oid} A shape={img.shape}, dtype={img.dtype}")
        env.close()


if __name__ == "__main__":
    main()
