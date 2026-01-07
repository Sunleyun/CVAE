# train_highlight.py
import os
import json
import argparse
import traceback
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from models.cvae import HighlightCVAE

# module logger (configured in main)
logger = logging.getLogger("train")


# ---------- Utils ----------
def decode_png_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode failed (bytes)")
    return img


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = img[:, :, None]
    return img.transpose(2, 0, 1)


def to_float01(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(hwc_to_chw(img)).contiguous()
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    if x.dtype == torch.uint16:
        return x.float() / 65535.0
    return x.float()


def to_mask01(img: np.ndarray) -> torch.Tensor:
    x = to_float01(img)
    if x.shape[0] != 1:
        x = x[:1]
    return x


def tensor_stats(x: torch.Tensor):
    x_ = x.detach()
    return {
        "shape": tuple(x_.shape),
        "min": float(torch.nan_to_num(x_.min()).item()),
        "max": float(torch.nan_to_num(x_.max()).item()),
        "mean": float(torch.nan_to_num(x_.mean()).item()),
        "dtype": str(x_.dtype),
    }


def save_debug_grid5(out_path: str, D, A, M_gt, M_pred, A_hat):
    """
    [D | A(gt) | M_gt | M_pred | A_hat]
    """
    def to_u8(x):
        x = torch.nan_to_num(x).clamp(0, 1)
        try:
            if x.shape[0] == 3:
                x = x.pow(1.0 / 2.2)
        except Exception:
            pass
        return (x * 255.0).byte().cpu().numpy()

    D_u8 = to_u8(D)
    A_u8 = to_u8(A)
    H_u8 = to_u8(A_hat)
    Mgt_u8 = to_u8(M_gt.repeat(3, 1, 1))
    Mpd_u8 = to_u8(M_pred.repeat(3, 1, 1))

    def chw_to_hwc(u8):
        return u8.transpose(1, 2, 0)

    canvas = np.concatenate(
        [chw_to_hwc(D_u8), chw_to_hwc(A_u8), chw_to_hwc(Mgt_u8), chw_to_hwc(Mpd_u8), chw_to_hwc(H_u8)],
        axis=1
    )
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------- Dataset ----------
class SHIQLmdbDataset(Dataset):
    def __init__(self, lmdb_path: str, keys: Optional[List[str]] = None):
        assert os.path.exists(lmdb_path), f"lmdb not found: {lmdb_path}"
        self.lmdb_path = lmdb_path
        self._env = None

        if keys is None:
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
            with env.begin(write=False) as txn:
                meta_b = txn.get(b"__meta__")
                if meta_b is None:
                    raise RuntimeError("LMDB missing __meta__")
                meta = json.loads(meta_b.decode("utf-8"))
                keys = meta.get("keys", [])
            env.close()

        if not keys:
            raise RuntimeError("No keys found.")
        self.keys = keys
        logger.info(f"[DATASET] keys={len(self.keys)}")

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
            )
        return self._env

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        oid = self.keys[idx]
        try:
            env = self._get_env()
            with env.begin(write=False) as txn:
                def get_img(suf):
                    b = txn.get(f"{oid}/{suf}".encode("utf-8"))
                    if b is None:
                        raise KeyError(f"missing key: {oid}/{suf}")
                    return decode_png_bytes(b)

                A = get_img("A")  # highlight
                D = get_img("D")  # no highlight
                S = get_img("S")  # soft mask
                T = get_img("T")  # binary mask

            A_t = to_float01(A)
            D_t = to_float01(D)
            S_t = to_mask01(S)
            T_t = (to_mask01(T) > 0.5).float()

            # Linearize RGB from sRGB to linear light for training stability
            try:
                gamma = 2.2
                A_lin = torch.pow(A_t, gamma)
                D_lin = torch.pow(D_t, gamma)
            except Exception:
                A_lin = A_t
                D_lin = D_t

            return {"id": oid, "A": A_lin, "D": D_lin, "S": S_t, "T": T_t}

        except Exception as e:
            logger.error(f"[GETITEM_ERROR] idx={idx} oid={oid} err={e}")
            logger.exception(traceback.format_exc())
            raise


def collate_keep_ids(batch):
    ids = [b["id"] for b in batch]
    A = torch.stack([b["A"] for b in batch], dim=0)
    D = torch.stack([b["D"] for b in batch], dim=0)
    S = torch.stack([b["S"] for b in batch], dim=0)
    T = torch.stack([b["T"] for b in batch], dim=0)
    return {"id": ids, "A": A, "D": D, "S": S, "T": T}


# ---------- Loss helpers ----------
def kl_divergence(mu, logvar):
    logvar = torch.clamp(logvar, -20.0, 10.0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def make_binary_from_S(S, thr: float):
    return (S > thr).float()


def safe_isfinite(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def split_mask_l1(R_hat, R_gt, M, bg_weight=0.05, fg_weight=10.0, eps=1e-6):
    # M: (B,1,H,W) in [0,1]
    # Compute per-pixel weighting: W = M*fg_weight + (1-M)*bg_weight
    M3 = M.repeat(1, 3, 1, 1)
    diff = (R_hat - R_gt).abs()

    W = M3 * fg_weight + (1.0 - M3) * bg_weight
    num = (diff * W).sum()
    den = W.sum().clamp_min(eps)
    return num / den


def resolve_z_dims(z_dim, z_intensity_dim, z_shape_dim, z_texture_dim):
    if z_intensity_dim is None and z_shape_dim is None and z_texture_dim is None:
        base = z_dim // 3
        rem = z_dim - base * 3
        dims = [base + (1 if rem > 0 else 0), base + (1 if rem > 1 else 0), base]
        return dims, z_dim
    if z_intensity_dim is None or z_shape_dim is None or z_texture_dim is None:
        raise ValueError("z_intensity_dim, z_shape_dim, z_texture_dim must be all set or all None")
    dims = [int(z_intensity_dim), int(z_shape_dim), int(z_texture_dim)]
    return dims, sum(dims)


def rgb_to_luminance(x):
    if x.shape[1] == 1:
        return x
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def masked_mean_std(x, mask, eps=1e-6):
    w = mask
    w_sum = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean = (x * w).sum(dim=(2, 3), keepdim=True) / w_sum
    var = ((x - mean) ** 2 * w).sum(dim=(2, 3), keepdim=True) / w_sum
    std = torch.sqrt(var + eps)
    return torch.cat([mean, std], dim=1).flatten(1)


def laplacian_map(x):
    kernel = x.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
    y = F.conv2d(x, kernel, padding=1)
    return torch.abs(y)


def fft_highpass_map(x, high_freq_ratio=0.5):
    _, _, h, w = x.shape
    fft = torch.fft.fft2(x.squeeze(1), norm="ortho")
    fy = torch.fft.fftfreq(h, d=1.0).to(x.device)
    fx = torch.fft.fftfreq(w, d=1.0).to(x.device)
    grid_y, grid_x = torch.meshgrid(fy, fx)
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    r_max = radius.max().clamp_min(1e-6)
    mask = (radius >= high_freq_ratio * r_max).to(fft.real.dtype)
    hp = torch.fft.ifft2(fft * mask, norm="ortho").real
    return hp.abs().unsqueeze(1)


def make_texture_target(x, method, high_freq_ratio):
    if method == "fft":
        return fft_highpass_map(x, high_freq_ratio)
    return laplacian_map(x)


# ---------- Train ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--outdir", default="./runs_highlight_residual")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--z_intensity_dim", type=int, default=None)
    ap.add_argument("--z_shape_dim", type=int, default=None)
    ap.add_argument("--z_texture_dim", type=int, default=None)
    ap.add_argument("--spec_map_hw", type=int, default=25)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--kl_anneal_steps", type=int, default=2000, help="steps to anneal KL weight (0 disables)")

    ap.add_argument("--mask_supervision", choices=["S", "T"], default="S")
    ap.add_argument("--s_thr", type=float, default=0.25, help="binarize S for BCE target")

    # loss weights
    ap.add_argument("--lambda_mask", type=float, default=1.0)
    ap.add_argument("--lambda_res", type=float, default=1.0, help="residual regression weight")
    ap.add_argument("--lambda_auxA", type=float, default=0.2, help="optional A_hat reconstruction weight (small)")
    ap.add_argument("--lambda_intensity", type=float, default=0.0, help="z_intensity -> brightness stats")
    ap.add_argument("--lambda_shape", type=float, default=0.0, help="z_shape -> mask shape")
    ap.add_argument("--lambda_texture", type=float, default=0.0, help="z_texture -> high-frequency")
    ap.add_argument("--texture_feat", choices=["laplacian", "fft"], default="laplacian")
    ap.add_argument("--texture_hf_ratio", type=float, default=0.5, help="high-freq cutoff ratio for FFT")

    # residual weighting (focus on highlight regions)
    ap.add_argument("--res_alpha", type=float, default=10.0, help="residual loss weight boost on mask")

    # mask teacher forcing schedule for conditioning
    ap.add_argument("--mask_warmup_steps", type=int, default=2000, help="0 disables")
    ap.add_argument("--detach_mask_in_comp", action="store_true", help="detach M_pred when composing/conditioning")

    # pos_weight stability
    ap.add_argument("--max_pos_weight", type=float, default=500.0, help="cap for BCE pos_weight to avoid extremes")
    ap.add_argument("--pos_momentum", type=float, default=0.9, help="EMA momentum for pos_ratio smoothing (0 disables)")

    # stability
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_debug_mode", choices=["epoch", "step", "off"], default="epoch",
                    help="debug save strategy")
    ap.add_argument("--save_debug_every", type=int, default=1000,
                    help="step interval when save_debug_mode=step")
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N samples (0=all)")
    ap.add_argument("--resume", default=None, help="path to checkpoint to resume training from")
    args = ap.parse_args()

    if args.spec_map_hw <= 0:
        raise ValueError("spec_map_hw must be positive")

    z_dims, z_total = resolve_z_dims(
        args.z_dim, args.z_intensity_dim, args.z_shape_dim, args.z_texture_dim
    )
    args.z_dim = z_total

    outdir = Path(args.outdir)
    (outdir / "debug").mkdir(parents=True, exist_ok=True)

    args_dump = dict(vars(args))
    args_dump["z_intensity_dim_resolved"] = z_dims[0]
    args_dump["z_shape_dim_resolved"] = z_dims[1]
    args_dump["z_texture_dim_resolved"] = z_dims[2]
    try:
        with open(outdir / "train_args.json", "w", encoding="utf-8") as f:
            json.dump(args_dump, f, ensure_ascii=True, indent=2)
    except Exception:
        logger.exception("[ARGS_SAVE] failed to write train_args.json")

    # configure logging: detailed -> file, quiet -> console
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(outdir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)

    # load keys
    env = lmdb.open(args.lmdb, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
    with env.begin(write=False) as txn:
        meta_b = txn.get(b"__meta__")
        if meta_b is None:
            raise RuntimeError("LMDB missing __meta__")
        meta = json.loads(meta_b.decode("utf-8"))
        keys = meta.get("keys", [])
    env.close()
    if not keys:
        raise RuntimeError("No keys in meta.")
    if args.limit and args.limit > 0:
        keys = keys[:args.limit]

    n_total = len(keys)
    n_train = int(n_total * args.train_ratio)
    train_keys = keys[:n_train]
    logger.info(f"[SPLIT] total={n_total} train={len(train_keys)} sup={args.mask_supervision}")

    train_ds = SHIQLmdbDataset(args.lmdb, keys=train_keys)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_keep_ids
    )

    model = HighlightCVAE(
        z_dim=args.z_dim,
        z_intensity_dim=z_dims[0],
        z_shape_dim=z_dims[1],
        z_texture_dim=z_dims[2],
        spec_map_hw=args.spec_map_hw,
        base_cvae=64,
        base_mask=32,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info(f"[INIT] device={args.device} amp={use_amp} beta={args.beta} "
                f"lambda_mask={args.lambda_mask} lambda_res={args.lambda_res} lambda_auxA={args.lambda_auxA} "
                f"lambda_intensity={args.lambda_intensity} lambda_shape={args.lambda_shape} lambda_texture={args.lambda_texture} "
                f"detach_mask_in_comp={args.detach_mask_in_comp} texture_feat={args.texture_feat} "
                f"save_debug_mode={args.save_debug_mode} save_debug_every={args.save_debug_every}")
    logger.info(f"[Z_DIM] z_dim={args.z_dim} intensity={z_dims[0]} shape={z_dims[1]} "
                f"texture={z_dims[2]} spec_map_hw={args.spec_map_hw}")

    global_step = 0
    use_spec = (args.lambda_intensity > 0) or (args.lambda_shape > 0) or (args.lambda_texture > 0)
    checked_spec_hw = False

    # EMA of positive ratio for stable pos_weight
    pos_ratio_ema = None

    # resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"[RESUME] checkpoint not found: {args.resume}")
            raise RuntimeError(f"Resume checkpoint not found: {args.resume}")
        logger.info(f"[RESUME] loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        incompatible = model.load_state_dict(ckpt.get("model", {}), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(f"[RESUME] missing_keys={len(incompatible.missing_keys)} "
                           f"unexpected_keys={len(incompatible.unexpected_keys)}")
        if "opt" in ckpt and ckpt["opt"] is not None:
            try:
                opt.load_state_dict(ckpt["opt"])
                for state in opt.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(args.device)
            except Exception:
                logger.exception("[RESUME] failed to load optimizer state; continuing without optimizer restore")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        logger.info(f"[RESUME] start_epoch={start_epoch} global_step={global_step}")

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs

    progress = None
    if tqdm is not None:
        try:
            progress = tqdm(total=total_steps, ncols=80, ascii=True, leave=True)
        except Exception:
            progress = None

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"[EPOCH] {epoch+1}/{args.epochs}")
        model.train()
        saved_debug_this_epoch = False

        for batch in train_dl:
            ids = batch["id"]
            A = batch["A"].to(args.device, non_blocking=True)
            D = batch["D"].to(args.device, non_blocking=True)
            if not checked_spec_hw:
                expected_hw = A.shape[-1] // 8
                if expected_hw != args.spec_map_hw:
                    logger.warning(f"[SPEC_MAP] spec_map_hw={args.spec_map_hw} expected={expected_hw} from input size")
                checked_spec_hw = True

            if args.mask_supervision == "T":
                M_gt = batch["T"].to(args.device, non_blocking=True)
                M_bin = M_gt
            else:
                M_gt = batch["S"].to(args.device, non_blocking=True)
                M_bin = make_binary_from_S(M_gt, args.s_thr)

            A_f32 = A.float()
            D_f32 = D.float()
            R_gt = torch.clamp(A_f32 - D_f32, 0.0, 1.0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                R_hat, M_logits, mu, logvar, spec_pred = model(D, A_gt=A, M_used=None)

            M_logits_f32 = M_logits.float()
            M_pred = torch.sigmoid(M_logits_f32)

            with torch.no_grad():
                pos = M_bin.sum().clamp_min(1.0)
                neg = (1.0 - M_bin).sum().clamp_min(1.0)
                batch_pos_ratio = (pos / (pos + neg)).item()
                if args.pos_momentum and args.pos_momentum > 0:
                    if pos_ratio_ema is None:
                        pos_ratio_ema = batch_pos_ratio
                    else:
                        pos_ratio_ema = args.pos_momentum * pos_ratio_ema + (1.0 - args.pos_momentum) * batch_pos_ratio
                    pos_ratio = float(pos_ratio_ema)
                else:
                    pos_ratio = batch_pos_ratio
                pos_weight_raw = (1.0 - pos_ratio) / max(1e-6, pos_ratio)
                pos_weight = float(min(max(1.0, pos_weight_raw), args.max_pos_weight))

            bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))
            mloss = bce(M_logits_f32, M_bin.float())

            if args.mask_warmup_steps and args.mask_warmup_steps > 0:
                p = min(1.0, global_step / float(args.mask_warmup_steps))
                M_used = (1.0 - p) * M_gt.float() + p * M_pred
            else:
                M_used = M_pred

            if args.detach_mask_in_comp:
                M_comp = M_used.detach()
            else:
                M_comp = M_used

            rloss = split_mask_l1(R_hat.float(), R_gt, M_gt.float(), bg_weight=0.05, fg_weight=args.res_alpha)

            A_hat = torch.clamp(D_f32 + R_hat * M_comp, 0.0, 1.0)
            auxA = F.l1_loss(A_hat, A_f32)

            klloss = kl_divergence(mu.float(), logvar.float())

            if args.kl_anneal_steps and args.kl_anneal_steps > 0:
                beta_eff = args.beta * min(1.0, global_step / float(args.kl_anneal_steps))
            else:
                beta_eff = args.beta

            loss = args.lambda_res * rloss + args.lambda_mask * mloss + args.lambda_auxA * auxA + beta_eff * klloss
            intensity_loss = None
            shape_loss = None
            texture_loss = None
            if use_spec:
                lum = rgb_to_luminance(R_gt)
                if args.lambda_intensity > 0:
                    intensity_target = masked_mean_std(lum, M_gt.float())
                    intensity_loss = F.l1_loss(spec_pred["intensity"].float(), intensity_target)
                    loss = loss + args.lambda_intensity * intensity_loss
                if args.lambda_shape > 0:
                    shape_target = F.interpolate(
                        M_bin.float(),
                        size=(args.spec_map_hw, args.spec_map_hw),
                        mode="bilinear",
                        align_corners=False,
                    )
                    shape_loss = F.binary_cross_entropy_with_logits(
                        spec_pred["shape_logits"].float(), shape_target
                    )
                    loss = loss + args.lambda_shape * shape_loss
                if args.lambda_texture > 0:
                    texture_target = make_texture_target(
                        lum, args.texture_feat, args.texture_hf_ratio
                    )
                    texture_target = F.interpolate(
                        texture_target,
                        size=(args.spec_map_hw, args.spec_map_hw),
                        mode="bilinear",
                        align_corners=False,
                    )
                    texture_loss = F.l1_loss(spec_pred["texture_map"].float(), texture_target)
                    loss = loss + args.lambda_texture * texture_loss

            if not safe_isfinite(loss):
                logger.warning(f"[WARN] non-finite loss at step={global_step}, skipping. "
                               f"rl={rloss.item()} ml={mloss.item()} auxA={auxA.item()} kl={klloss.item()}")
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    msg = (f"[STEP {global_step}] loss={loss.item():.4f} "
                           f"res={rloss.item():.4f} mask={mloss.item():.4f} "
                           f"auxA={auxA.item():.4f} kl={klloss.item():.4f}")
                    if intensity_loss is not None:
                        msg += f" int={intensity_loss.item():.4f}"
                    if shape_loss is not None:
                        msg += f" shape={shape_loss.item():.4f}"
                    if texture_loss is not None:
                        msg += f" tex={texture_loss.item():.4f}"
                    msg += f" id0={ids[0]}"
                    logger.info(msg)
                    logger.info(f"  pos_ratio={pos_ratio:.6f} pos_weight={pos_weight:.2f} s_thr={args.s_thr} "
                                f"p_warm={min(1.0, global_step/max(1,args.mask_warmup_steps)) if args.mask_warmup_steps else 1.0:.3f}")
                    logger.info(f"  M_gt    {tensor_stats(M_gt.float())}")
                    logger.info(f"  M_logits{tensor_stats(M_logits_f32)}")
                    logger.info(f"  M_pred  {tensor_stats(M_pred)}")
                    logger.info(f"  R_gt    {tensor_stats(R_gt)}")
                    logger.info(f"  R_hat   {tensor_stats(R_hat.float())}")
                    logger.info(f"  A_hat   {tensor_stats(A_hat)}")

            save_debug = False
            if args.save_debug_mode == "epoch":
                if not saved_debug_this_epoch:
                    save_debug = True
                    saved_debug_this_epoch = True
            elif args.save_debug_mode == "step":
                if args.save_debug_every > 0 and global_step % args.save_debug_every == 0:
                    save_debug = True

            if save_debug:
                with torch.no_grad():
                    D0 = D_f32[0].detach()
                    A0 = A_f32[0].detach()
                    Mgt0 = M_gt[0].float().detach()
                    Mp0 = M_pred[0].detach()
                    Ahat0 = A_hat[0].detach()
                    pth = outdir / "debug" / f"e{epoch+1:02d}_s{global_step:07d}_{ids[0]}_sup{args.mask_supervision}.png"
                    save_debug_grid5(str(pth), D0, A0, Mgt0, Mp0, Ahat0)
                    logger.info(f"[SAVE_DEBUG] {pth}")

            global_step += 1
            if progress is not None:
                try:
                    progress.update(1)
                except Exception:
                    pass
            else:
                try:
                    print(f"\r[PROG] step {global_step}/{total_steps} (epoch {epoch+1}/{args.epochs})", end="", flush=True)
                except Exception:
                    pass

        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        }
        ckpt_path = outdir / f"ckpt_epoch{epoch+1:02d}_sup{args.mask_supervision}.pt"
        torch.save(ckpt, ckpt_path)
        logger.info(f"[SAVE_CKPT] {ckpt_path}")

    try:
        if progress is not None:
            progress.close()
        else:
            print("")
    except Exception:
        pass
    logger.info("[DONE] training finished.")


if __name__ == "__main__":
    main()
