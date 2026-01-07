import os
import subprocess

"""
Simple runner for training with saved default parameters.
Modify `DEFAULT` below to change the training config.
"""
import shlex

DEFAULT = {
    "lmdb": "./tools/shiQ_train.lmdb",
    "outdir": "./runs_cvae_debug_T_0",
    "device": None,
    "batch_size": 16,
    "num_workers": 4,
    "epochs": 50,
    "lr": 2e-4,
    "z_dim": 32,
    "z_intensity_dim": None,
    "z_shape_dim": None,
    "z_texture_dim": None,
    "spec_map_hw": 25,
    "beta": 0.01,
    "kl_anneal_steps": 2000,
    "mask_supervision": "S",
    "s_thr": 0.25,
    "lambda_mask": 2.0,
    "lambda_res": 1.0,
    "lambda_auxA": 0.2,
    "lambda_intensity": 0.2,
    "lambda_shape": 0.5,
    "lambda_texture": 0.2,
    "texture_feat": "laplacian",
    "texture_hf_ratio": 0.5,
    "res_alpha": 15,
    "mask_warmup_steps": 2000,
    "detach_mask_in_comp": True,
    "max_pos_weight": 500.0,
    "pos_momentum": 0.9,
    "amp": False,
    "grad_clip": 1.0,
    "log_every": 10,
    "save_debug_mode": "epoch",
    "save_debug_every": 1000,
    "train_ratio": 0.98,
    "limit": 0,
    "resume": None,
}

def build_cmd(cfg: dict):
    cmd = ["python", "train.py"]
    for k, v in cfg.items():
        if v is None or v == "":
            continue
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(key)
        else:
            cmd.extend([key, str(v)])
    return cmd

if __name__ == "__main__":
    cmd = build_cmd(DEFAULT)
    print("Running short training with cmd:")
    print(shlex.join(cmd))
    subprocess.run(cmd)
