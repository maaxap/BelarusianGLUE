# Borrowed with modifications from https://www.llama.com/docs/how-to-guides/fine-tuning

from sys import argv
import os

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

assert len(argv) == 3, "Usage: python3 merge_lora <model_path> <adapter_path>"
model_path, adapter_path = argv[1:]

checkpoint_dirs = sorted(
    [x for x in os.listdir(adapter_path) if x.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)
assert len(checkpoint_dirs) == 1
last_checkpoint = os.path.join(adapter_path, checkpoint_dirs[-1])

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, last_checkpoint)
print("Merging adapter %s with the base model %s..." % (last_checkpoint, model_path))
model = model.merge_and_unload()
output_path = os.path.join(adapter_path, "merged")
print("Saving as %s..." % output_path)
model.save_pretrained(output_path)
print("Done!")
