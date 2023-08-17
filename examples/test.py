import argparse
import json
import math
import os
import sys
import time


from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from compressai.models.waseda import Cheng2020Attention
from torchvision import transforms
from compressai.zoo import load_state_dict
from glob import glob
from torchvision import transforms
unloader = transforms.ToPILImage()

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def inference(model, x):
    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }

def infer_likehood(model, x):

    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    out_enc = model(x_padded)

    out_enc["x_hat"] = F.pad(
        out_enc["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )
    
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    
    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_enc["likelihoods"].values()).detach().cpu().item()




    return {
        "psnr": psnr(x, out_enc["x_hat"]),
        "ms-ssim": ms_ssim(x, out_enc["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
    } 

def eval_model(model, filepaths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = defaultdict(float)
    for f in filepaths:
        x = read_image(f).to(device)
        model = model.to(device)
        x = x
        rv = infer_likehood(model, x)
        print(rv)

        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics

if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datapath = r"./examples/cheng2020_attn-mse-1-465f2b64.pth"


    checkpoint_dict = torch.load(datapath)
    net = Cheng2020Attention.from_state_dict(checkpoint_dict).eval().to(device)

    test_path="./Kodak/*.png"
    net.update(force=True)
    q=eval_model(net,glob(test_path))
    print(q)