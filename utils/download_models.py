# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
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
    print("Downloading Amphion Singing BigVGAN...")
    dl_model(
        "https://huggingface.co/amphion/BigVGAN_singing_bigdata/resolve/main/bigvgan_singing/400000.pt",
        BASE_DIR / "pretrained/bigvgan/400000.pt")
    dl_model(
        "https://huggingface.co/amphion/BigVGAN_singing_bigdata/resolve/main/bigvgan_singing/args.json",
        BASE_DIR / "pretrained/bigvgan/args.json")

    print("Downloading Amphion Speech HiFi-GAN...")
    if not os.path.exists(BASE_DIR / "pretrained/hifigan/hifigan_speech/args.json"):
        snapshot_download("amphion/hifigan_speech_bigdata", local_dir=BASE_DIR / "pretrained/hifigan",
                          ignore_patterns="README.md")

    print("Downloading Amphion DiffWave...")
    if not os.path.exists(BASE_DIR / "pretrained/diffwave/diffwave_speech/args.json"):
        snapshot_download("amphion/diffwave", local_dir=BASE_DIR / "pretrained/diffwave",
                          ignore_patterns="README.md")
        os.rename(BASE_DIR / "pretrained/diffwave/diffwave", BASE_DIR / "pretrained/diffwave/diffwave_speech")

    print("Downloading ContentVec...")
    dl_model(
        "https://huggingface.co/LukeJacob2023/ContentVec/resolve/main/checkpoint_best_legacy_500.pt",
        BASE_DIR / "pretrained/contentvec/checkpoint_best_legacy_500.pt")

    print("Downloading Whisper...")
    dl_model(
        "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        BASE_DIR / "pretrained/whisper/medium.pt")

    print("Downloading RawNet3...")
    dl_model(
        "https://huggingface.co/jungjee/RawNet3/resolve/main/model.pt",
        BASE_DIR / "pretrained/rawnet3/model.pt")
    print("All models downloaded!")
