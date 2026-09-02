from __future__ import annotations
import hashlib, random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def image_sha256(path: str|Path) -> str:
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def preprocessing_seed(path: str|Path) -> int: return int(image_sha256(path)[:16],16) % (2**32)

@contextmanager
def preserved_rng(seed: int):
 py=random.getstate(); npstate=np.random.get_state(); tcpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 try: yield
 finally:
  random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(tcpu)
  if cuda is not None: torch.cuda.set_rng_state_all(cuda)

def deterministic_process_image(path: str|Path, image_processor, model_config):
 from llava.mm_utils import process_images
 seed=preprocessing_seed(path)
 with preserved_rng(seed):
  image=Image.open(path).convert("RGB"); tensor=process_images([image],image_processor,model_config)
 return tensor,seed,image_sha256(path)
