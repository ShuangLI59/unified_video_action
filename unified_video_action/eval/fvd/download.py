import requests
from tqdm import tqdm
import os
import torch

from utils import download


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit="iB", unit_scale=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


_I3D_PRETRAINED_ID = "1fBNl3TS0LA5FEhZv5nMGJs2_7qQmvTmh"


def load_i3d_pretrained(device=torch.device("cpu")):
    from evals.fvd.pytorch_i3d import InceptionI3d

    i3d = InceptionI3d(400, in_channels=3).to(device)
    # filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    filepath = "i3d_pretrained_400.pt"
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    return i3d
