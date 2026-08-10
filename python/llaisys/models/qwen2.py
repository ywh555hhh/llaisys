from typing import Sequence
from ..libllaisys import DeviceType

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        if device == DeviceType.NVIDIA:
            self.torch_device = torch.device("cuda:0")
        else:
            self.torch_device = torch.device("cpu")
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.torch_device,
                trust_remote_code=True,
            )
            self._model.eval()
        return self._model

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        model = self._load_model()
        input_ids = torch.tensor([list(inputs)], dtype=torch.long, device=model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        return outputs[0].tolist()
