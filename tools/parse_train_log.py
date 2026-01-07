import re
import csv
from pathlib import Path

log_path = Path('runs_cvae_debug_S_long/train.log')
out_csv = Path('runs_cvae_debug_S_long/epoch_metrics.csv')

pattern_step = re.compile(r"\[STEP\s+\d+\]\s+loss=([0-9\.eE+-]+)\s+res=([0-9\.eE+-]+)\s+mask=([0-9\.eE+-]+)\s+auxA=([0-9\.eE+-]+)\s+kl=([0-9\.eE+-]+)")
pattern_pos = re.compile(r"pos_ratio=([0-9\.eE+-]+)\s+pos_weight=([0-9\.eE+-]+)")
pattern_epoch = re.compile(r"\[EPOCH\]\s+(\d+)/(\d+)")
pattern_save = re.compile(r"\[SAVE_CKPT\].*ckpt_epoch(\d+)_supS\.pt")

# We'll aggregate by checkpoint epoch: when a [SAVE_CKPT] appears record the last seen STEP metrics before it.

last_step_values = None
rows = []
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern_step.search(line)
        if m:
            loss, res, mask, auxA, kl = m.groups()
            last_step_values = {
                'loss': float(loss), 'res': float(res), 'mask': float(mask), 'auxA': float(auxA), 'kl': float(kl)
            }
        m2 = pattern_pos.search(line)
        if m2 and last_step_values is not None:
            pos_ratio, pos_weight = m2.groups()
            last_step_values['pos_ratio'] = float(pos_ratio)
            last_step_values['pos_weight'] = float(pos_weight)
        m3 = pattern_save.search(line)
        if m3 and last_step_values is not None:
            epoch = int(m3.group(1))
            row = {'epoch': epoch}
            row.update(last_step_values)
            rows.append(row)
            last_step_values = None

# write CSV sorted by epoch
rows = sorted(rows, key=lambda r: r['epoch'])
with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['epoch','loss','res','mask','auxA','kl','pos_ratio','pos_weight'])
    for r in rows:
        writer.writerow([r.get('epoch'), r.get('loss'), r.get('res'), r.get('mask'), r.get('auxA'), r.get('kl'), r.get('pos_ratio'), r.get('pos_weight')])

print(f"Wrote {len(rows)} rows to {out_csv}")
