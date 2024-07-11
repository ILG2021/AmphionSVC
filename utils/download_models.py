# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import zipfile
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent.parent


def dl_model(link, save_path):
    # Don't download repeat
    if not os.path.exists(save_path):
        with requests.get(link) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


if __name__ == "__main__":
    print("Downloading OpenVPI nsf hifigan...")
    if not os.path.exists("pretrained/nsf_hifigan/model.ckpt"):
        dl_model(
            "https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip",
            "./pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip")
        # 解压文件到指定目录
        with zipfile.ZipFile("./pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip", "r") as zip_ref:
            zip_ref.extractall("./pretrained")
        shutil.move("./pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02", "./pretrained/nsf_hifigan")

    print("Downloading ContentVec...")
    dl_model(
        "https://huggingface.co/LukeJacob2023/ContentVec/resolve/main/checkpoint_best_legacy_500.pt",
        BASE_DIR / "pretrained/contentvec/checkpoint_best_legacy_500.pt")

    print("Downloading Whisper...")
    dl_model(
        "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        BASE_DIR / "pretrained/whisper/large-v2.pt")
