import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from accelerate import dispatch_model, infer_auto_device_map
from peft.utils import PeftType, set_peft_model_state_dict
import copy
import transformers
import json
import warnings
import os
import torch.distributed as dist
from typing import Optional, Tuple, Union, List, Callable
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationConfig,
    GenerationMixin,
)
from torch import nn
from transformers import  GenerationConfig
from stream_generation import SteamGenerationMixin

class PeftCausalLMForSteamGeneration(PeftModelForCausalLM, SteamGenerationMixin):
    # support for streamly generation
    # default it call `model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)`, not cls!! so inherent PeftModelForCausalLM is no sense
    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        # load the config
        config = LoraConfig.from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        # here is the hack
        model = cls(model, config)
        model._reorder_cache = model.base_model._reorder_cache

        # load weights if any
        if os.path.exists(os.path.join(model_id, "adapter_model.bin")):
            filename = os.path.join(model_id, "adapter_model.bin")
        else:
            try:
                filename = hf_hub_download(model_id, "adapter_model.bin")
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {'adapter_model.bin'} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model

