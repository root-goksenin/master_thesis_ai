from enum import Enum, auto
import torch

class QueryAugmentMod(Enum):
  None_ = "None"
  UseNew = "Retrive_New"
  UsePast = "Use_Past"


def generate(model, encodings, augment_per_query: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None, device: str = "cuda"):
    with torch.no_grad():
      if not temperature:
          outs = model.generate(
              input_ids=encodings['input_ids'].to(device),
              do_sample=True,
              max_length=max_length,  # 64
              top_k=top_k,  # 25
              top_p=top_p,  # 0.95
              num_return_sequences=augment_per_query  # 1
              )
      else:
          outs = model.generate(
              input_ids=encodings['input_ids'].to(device),
              do_sample=True,
              max_length=max_length,  # 64
              top_k=top_k,  # 25
              temperature=temperature,
              num_return_sequences=augment_per_query  # 1
              )
    return outs