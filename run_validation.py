import torch
from pathlib import Path
import lmdb, json, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader

from models.cvae import HighlightCVAE

# minimal helpers (copied from train.py)

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
    if x.dtype == np.uint8:
        return x.float() / 255.0
    if x.dtype == np.uint16:
        return x.float() / 65535.0
    return x.float()


def to_mask01(img: np.ndarray) -> torch.Tensor:
    x = to_float01(img)
    if x.shape[0] != 1:
        x = x[:1]
    return x


class SHIQLmdbDataset(Dataset):
    def __init__(self, lmdb_path: str, keys: list = None, limit: int = 0):
        self.lmdb_path = lmdb_path
        self._env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=True)
        with self._env.begin(write=False) as txn:
            meta_b = txn.get(b"__meta__")
            meta = json.loads(meta_b.decode('utf-8'))
            keys_all = meta.get('keys', [])
        if limit and limit > 0:
            keys_all = keys_all[:limit]
        self.keys = keys_all

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        oid = self.keys[idx]
        with self._env.begin(write=False) as txn:
            def get_img(suf):
                b = txn.get(f"{oid}/{suf}".encode('utf-8'))
                return decode_png_bytes(b)
            A = get_img('A')
            D = get_img('D')
            S = get_img('S')
            T = get_img('T')
        A_t = to_float01(A)
        D_t = to_float01(D)
        S_t = to_mask01(S)
        T_t = (to_mask01(T) > 0.5).float()
        try:
            gamma = 2.2
            A_lin = torch.pow(A_t, gamma)
            D_lin = torch.pow(D_t, gamma)
        except Exception:
            A_lin = A_t
            D_lin = D_t
        return {'id': oid, 'A': A_lin, 'D': D_lin, 'S': S_t, 'T': T_t}


def chw_to_hwc(u8):
    return u8.transpose(1,2,0)


def save_debug_grid5(out_path: str, D, A, M_gt, M_pred, A_hat):
    def to_u8(x):
        x = torch.nan_to_num(x).clamp(0,1)
        try:
            if x.shape[0] == 3:
                x = x.pow(1.0/2.2)
        except Exception:
            pass
        return (x*255.0).byte().cpu().numpy()
    D_u8 = to_u8(D)
    A_u8 = to_u8(A)
    H_u8 = to_u8(A_hat)
    Mgt_u8 = to_u8(M_gt.repeat(3,1,1))
    Mpd_u8 = to_u8(M_pred.repeat(3,1,1))
    canvas = np.concatenate([chw_to_hwc(D_u8), chw_to_hwc(A_u8), chw_to_hwc(Mgt_u8), chw_to_hwc(Mpd_u8), chw_to_hwc(H_u8)], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    lmdb_path = Path('tools/shiQ_train.lmdb')
    ckpt = Path('runs_cvae_debug_S_long/ckpt_epoch120_supS.pt')
    outdir = Path('runs_cvae_debug_S_long/validation')
    outdir.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = SHIQLmdbDataset(str(lmdb_path), limit=10)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = HighlightCVAE(z_dim=32, base_cvae=64, base_mask=32).to(device)
    if not ckpt.exists():
        print('Checkpoint not found:', ckpt)
        raise SystemExit(1)
    ck = torch.load(str(ckpt), map_location=device)
    incompatible = model.load_state_dict(ck.get('model', {}), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[LOAD] missing_keys={len(incompatible.missing_keys)} "
              f"unexpected_keys={len(incompatible.unexpected_keys)}")
    model.eval()

    tot_l1 = 0.0
    n = 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if i >= 10:
                break
            A = b['A'].to(device)
            D = b['D'].to(device)
            S = b['S'].to(device)
            T = b['T'].to(device)

            R_hat, M_logits, mu, logvar, _ = model(D, A_gt=None, M_used=None)
            M_pred = torch.sigmoid(M_logits)
            A_hat = (D + R_hat * M_pred).clamp(0,1)

            l1 = (A_hat - A).abs().mean().item()
            tot_l1 += l1
            n += 1

            out_path = outdir / f'val_{i:02d}.png'
            save_debug_grid5(str(out_path), D[0].cpu(), A[0].cpu(), (S[0]>0.5).float().cpu(), M_pred[0].cpu(), A_hat[0].cpu())
            print(f'saved {out_path} l1={l1:.6f}')

    if n:
        print('avg_l1=', tot_l1 / n)
    else:
        print('no samples')
