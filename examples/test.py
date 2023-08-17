import argparse
import json
import math
import os
import sys
import time
from torch.hub import load_state_dict_from_url


from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from compressai.models.waseda import Cheng2020Attention
from compressai.zoo import cheng2020_attn
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
    #out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_enc["x_hat"] = F.pad(
        out_enc["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr": psnr(x, out_enc["x_hat"]),
        "ms-ssim": ms_ssim(x, out_enc["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }



def eval_model(model, filepaths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = defaultdict(float)
    for f in filepaths:
        x = read_image(f).to(device)
        model = model.to(device)
        x = x
        rv = inference(model, x)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def download_checkpoint(model_name):
    root_url = "https://compressai.s3.amazonaws.com/models/v1"
    model_urls = {
    "bmshj2018-factorized": {
        "mse": {
            1: f"{root_url}/bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
            2: f"{root_url}/bmshj2018-factorized-prior-2-87279a02.pth.tar",
            3: f"{root_url}/bmshj2018-factorized-prior-3-5c6f152b.pth.tar",
            4: f"{root_url}/bmshj2018-factorized-prior-4-1ed4405a.pth.tar",
            5: f"{root_url}/bmshj2018-factorized-prior-5-866ba797.pth.tar",
            6: f"{root_url}/bmshj2018-factorized-prior-6-9b02ea3a.pth.tar",
            7: f"{root_url}/bmshj2018-factorized-prior-7-6dfd6734.pth.tar",
            8: f"{root_url}/bmshj2018-factorized-prior-8-5232faa3.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/bmshj2018-factorized-ms-ssim-1-9781d705.pth.tar",
            2: f"{root_url}/bmshj2018-factorized-ms-ssim-2-4a584386.pth.tar",
            3: f"{root_url}/bmshj2018-factorized-ms-ssim-3-5352f123.pth.tar",
            4: f"{root_url}/bmshj2018-factorized-ms-ssim-4-4f91b847.pth.tar",
            5: f"{root_url}/bmshj2018-factorized-ms-ssim-5-b3a88897.pth.tar",
            6: f"{root_url}/bmshj2018-factorized-ms-ssim-6-ee028763.pth.tar",
            7: f"{root_url}/bmshj2018-factorized-ms-ssim-7-8c265a29.pth.tar",
            8: f"{root_url}/bmshj2018-factorized-ms-ssim-8-8811bd14.pth.tar",
        },
    },
    "bmshj2018-hyperprior": {
        "mse": {
            1: f"{root_url}/bmshj2018-hyperprior-1-7eb97409.pth.tar",
            2: f"{root_url}/bmshj2018-hyperprior-2-93677231.pth.tar",
            3: f"{root_url}/bmshj2018-hyperprior-3-6d87be32.pth.tar",
            4: f"{root_url}/bmshj2018-hyperprior-4-de1b779c.pth.tar",
            5: f"{root_url}/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
            6: f"{root_url}/bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
            7: f"{root_url}/bmshj2018-hyperprior-7-3804dcbd.pth.tar",
            8: f"{root_url}/bmshj2018-hyperprior-8-a583f0cf.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/bmshj2018-hyperprior-ms-ssim-1-5cf249be.pth.tar",
            2: f"{root_url}/bmshj2018-hyperprior-ms-ssim-2-1ff60d1f.pth.tar",
            3: f"{root_url}/bmshj2018-hyperprior-ms-ssim-3-92dd7878.pth.tar",
            4: f"{root_url}/bmshj2018-hyperprior-ms-ssim-4-4377354e.pth.tar",
            5: f"{root_url}/bmshj2018-hyperprior-ms-ssim-5-c34afc8d.pth.tar",
            6: f"{root_url}/bmshj2018-hyperprior-ms-ssim-6-3a6d8229.pth.tar",
            7: f"{root_url}/bmshj2018-hyperprior-ms-ssim-7-8747d3bc.pth.tar",
            8: f"{root_url}/bmshj2018-hyperprior-ms-ssim-8-cc15b5f3.pth.tar",
        },
    },
    "mbt2018-mean": {
        "mse": {
            1: f"{root_url}/mbt2018-mean-1-e522738d.pth.tar",
            2: f"{root_url}/mbt2018-mean-2-e54a039d.pth.tar",
            3: f"{root_url}/mbt2018-mean-3-723404a8.pth.tar",
            4: f"{root_url}/mbt2018-mean-4-6dba02a3.pth.tar",
            5: f"{root_url}/mbt2018-mean-5-d504e8eb.pth.tar",
            6: f"{root_url}/mbt2018-mean-6-a19628ab.pth.tar",
            7: f"{root_url}/mbt2018-mean-7-d5d441d1.pth.tar",
            8: f"{root_url}/mbt2018-mean-8-8089ae3e.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/mbt2018-mean-ms-ssim-1-5bf9c0b6.pth.tar",
            2: f"{root_url}/mbt2018-mean-ms-ssim-2-e2a1bf3f.pth.tar",
            3: f"{root_url}/mbt2018-mean-ms-ssim-3-640ce819.pth.tar",
            4: f"{root_url}/mbt2018-mean-ms-ssim-4-12626c13.pth.tar",
            5: f"{root_url}/mbt2018-mean-ms-ssim-5-1be7f059.pth.tar",
            6: f"{root_url}/mbt2018-mean-ms-ssim-6-b83bf379.pth.tar",
            7: f"{root_url}/mbt2018-mean-ms-ssim-7-ddf9644c.pth.tar",
            8: f"{root_url}/mbt2018-mean-ms-ssim-8-0cc7b94f.pth.tar",
        },
    },
    "mbt2018": {
        "mse": {
            1: f"{root_url}/mbt2018-1-3f36cd77.pth.tar",
            2: f"{root_url}/mbt2018-2-43b70cdd.pth.tar",
            3: f"{root_url}/mbt2018-3-22901978.pth.tar",
            4: f"{root_url}/mbt2018-4-456e2af9.pth.tar",
            5: f"{root_url}/mbt2018-5-b4a046dd.pth.tar",
            6: f"{root_url}/mbt2018-6-7052e5ea.pth.tar",
            7: f"{root_url}/mbt2018-7-8ba2bf82.pth.tar",
            8: f"{root_url}/mbt2018-8-dd0097aa.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/mbt2018-ms-ssim-1-2878436b.pth.tar",
            2: f"{root_url}/mbt2018-ms-ssim-2-c41cb208.pth.tar",
            3: f"{root_url}/mbt2018-ms-ssim-3-d0dd64e8.pth.tar",
            4: f"{root_url}/mbt2018-ms-ssim-4-a120e037.pth.tar",
            5: f"{root_url}/mbt2018-ms-ssim-5-9b30e3b7.pth.tar",
            6: f"{root_url}/mbt2018-ms-ssim-6-f8b3626f.pth.tar",
            7: f"{root_url}/mbt2018-ms-ssim-7-16e6ff50.pth.tar",
            8: f"{root_url}/mbt2018-ms-ssim-8-0cb49d43.pth.tar",
        },
    },
    "cheng2020-anchor": {
        "mse": {
            1: f"{root_url}/cheng2020-anchor-1-dad2ebff.pth.tar",
            2: f"{root_url}/cheng2020-anchor-2-a29008eb.pth.tar",
            3: f"{root_url}/cheng2020-anchor-3-e49be189.pth.tar",
            4: f"{root_url}/cheng2020-anchor-4-98b0b468.pth.tar",
            5: f"{root_url}/cheng2020-anchor-5-23852949.pth.tar",
            6: f"{root_url}/cheng2020-anchor-6-4c052b1a.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/cheng2020_anchor-ms-ssim-1-20f521db.pth.tar",
            2: f"{root_url}/cheng2020_anchor-ms-ssim-2-c7ff5812.pth.tar",
            3: f"{root_url}/cheng2020_anchor-ms-ssim-3-c23e22d5.pth.tar",
            4: f"{root_url}/cheng2020_anchor-ms-ssim-4-0e658304.pth.tar",
            5: f"{root_url}/cheng2020_anchor-ms-ssim-5-c0a95e77.pth.tar",
            6: f"{root_url}/cheng2020_anchor-ms-ssim-6-f2dc1913.pth.tar",
        },
    },
    "cheng2020-attn": {
        "mse": {
            1: f"{root_url}/cheng2020_attn-mse-1-465f2b64.pth.tar",
            2: f"{root_url}/cheng2020_attn-mse-2-e0805385.pth.tar",
            3: f"{root_url}/cheng2020_attn-mse-3-2d07bbdf.pth.tar",
            4: f"{root_url}/cheng2020_attn-mse-4-f7b0ccf2.pth.tar",
            5: f"{root_url}/cheng2020_attn-mse-5-26c8920e.pth.tar",
            6: f"{root_url}/cheng2020_attn-mse-6-730501f2.pth.tar",
        },
        "ms-ssim": {
            1: f"{root_url}/cheng2020_attn-ms-ssim-1-c5381d91.pth.tar",
            2: f"{root_url}/cheng2020_attn-ms-ssim-2-5dad201d.pth.tar",
            3: f"{root_url}/cheng2020_attn-ms-ssim-3-5c9be841.pth.tar",
            4: f"{root_url}/cheng2020_attn-ms-ssim-4-8b2f647e.pth.tar",
            5: f"{root_url}/cheng2020_attn-ms-ssim-5-5ca1f34c.pth.tar",
            6: f"{root_url}/cheng2020_attn-ms-ssim-6-216423ec.pth.tar",
        },
    },
    }  
    
    target_model = model_urls["cheng2020-attn"]
    for (key_parent,value_dic) in target_model.items():
        for (key,value) in value_dic.items():
                state_dict = load_state_dict_from_url(value, progress=True)

if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datapath="./postmu/lambda0.01_best.pth"  #./lmu_0.01.pth.tar_best.pth.tar
    net = Cheng2020Attention(quality=2, pretrained=True).eval().to(device)

    test_path="./../../data/test/*.png"
    net.update(force=True)
    q=eval_model(net,glob(test_path))
    print(torch.load(datapath)["epoch"])
    print(q)