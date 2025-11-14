# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import inspect
import logging
import os
import re
from collections import OrderedDict
from typing import Optional

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    load_kwargs = {'map_location': 'cpu'}
    # PyTorch 2.6+ defaults `weights_only=True`, which breaks legacy .tar checkpoints.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False
    checkpoint = _load_with_zip_padding_fallback(model_pth, load_kwargs)
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(checkpoint, strict=True)
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def _load_with_zip_padding_fallback(model_pth: str, load_kwargs: dict) -> dict:
    """
    torch.load mis-detects checkpoints when zero padding precedes the PK zip header.
    Retry by seeking past the padding so PyTorch sees the correct signature.
    """
    try:
        return torch.load(model_pth, **load_kwargs)
    except (KeyError, RuntimeError) as err:
        if 'storages' not in str(err) and 'Damaged Zip archive' not in str(err):
            raise
        offset = _find_zip_signature_offset(model_pth)
        if offset is None:
            raise
        logging.warning("Detected padded zip checkpoint at %s; skipping first %d bytes", model_pth, offset)
        try:
            with _CheckpointFileView(model_pth, offset) as handle:
                return torch.load(handle, **load_kwargs)
        except Exception as zip_err:
            raise RuntimeError(
                f"Unable to load padded checkpoint {model_pth}. The archive appears corrupted; "
                "please re-download or verify the file."
            ) from zip_err


def _find_zip_signature_offset(model_pth: str, signature: bytes = b'PK\x03\x04',
                               chunk_size: int = 1024 * 1024) -> Optional[int]:
    """
    Locate the first PK zip header in a checkpoint file.
    Some distributed download tools pad files with zeros, confusing torch.load's zip detection.
    """
    overlap = len(signature) - 1
    prev = b''
    offset = 0
    with open(model_pth, 'rb') as src:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                return None
            data = prev + chunk
            idx = data.find(signature)
            if idx != -1:
                return offset - len(prev) + idx
            prev = data[-overlap:] if overlap > 0 else b''
            offset += len(chunk)


class _CheckpointFileView:
    """
    Presents a padded checkpoint as if it started at the provided offset.
    PyTorch's zip reader seeks using offsets relative to the logical start,
    so we intercept seek/tell to add the padding back transparently.
    """

    def __init__(self, model_pth: str, offset: int):
        self._path = model_pth
        self._offset = offset
        self._file = open(model_pth, 'rb')
        self._size = os.path.getsize(model_pth) - offset
        self.seek(0)

    def read(self, size: int = -1) -> bytes:
        return self._file.read(size)

    def seek(self, pos: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = self._offset + pos
        elif whence == os.SEEK_CUR:
            target = self._file.tell() + pos
        elif whence == os.SEEK_END:
            target = self._offset + self._size + pos
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        self._file.seek(target, os.SEEK_SET)
        return self.tell()

    def tell(self) -> int:
        return self._file.tell() - self._offset

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def readable(self) -> bool:  # for io.IOBase compatibility
        return True

    def seekable(self) -> bool:
        return True
