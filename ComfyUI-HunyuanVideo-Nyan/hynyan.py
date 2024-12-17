import os
import torch
from torch import Tensor, nn
import time 
import copy
import importlib
import folder_paths
import json
import gc
import re
from typing import List, Dict, Any, Tuple

import comfy
#import comfy.diffusers_load
#import comfy.clip_model
import comfy.sd
import comfy.utils
import comfy.supported_models_base
import comfy.model_management as mm
from comfy.model_management import get_torch_device
from comfy.utils import load_torch_file, save_torch_file
import comfy.model_base
import comfy.latent_formats
import node_helpers

from .utils import log, print_memory
from diffusers.video_processor import VideoProcessor
from .hyvideo.constants import PROMPT_TEMPLATE
from .hyvideo.text_encoder import TextEncoder
from .hyvideo.utils.data_utils import align_to
from .hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from .hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from .hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from .hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .hyvideo.modules.models import HYVideoDiffusionTransformer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
folder_paths.add_model_folder_path("hyvid_embeds", os.path.join(folder_paths.get_output_directory(), "hyvid_embeds"))
script_directory = os.path.dirname(os.path.abspath(__file__))

def filter_state_dict_by_blocks(state_dict, blocks_mapping):
    filtered_dict = {}

    for key in state_dict:
        if 'double_blocks.' in key or 'single_blocks.' in key:
            block_pattern = key.split('diffusion_model.')[1].split('.', 2)[0:2]
            block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

            if block_key in blocks_mapping:
                filtered_dict[key] = state_dict[key]

    return filtered_dict

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
        new_sd[k] = v
    return new_sd


class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.LatentFormat()
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        # Don't know what this is. Value taken from ComfyUI Mochi model.
        self.memory_usage_factor = 2.0
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True

# Original author: https://github.com/kijai/ComfyUI-HunyuanVideoWrapper
class HyVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_varlen",
                    "sageattn_varlen",
                    "comfy",
                    ], {"default": "flash_attn"}),
                "compile_args": ("COMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("HYVIDLORA", {"default": None}),
            }
        }

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None):
        transformer = None
        mm.unload_all_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn_varlen
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device)

        in_channels = out_channels = 16
        factor_kwargs = {"device": transformer_load_device, "dtype": base_dtype}
        HUNYUAN_VIDEO_CONFIG = {
            "mm_double_blocks_depth": 20,
            "mm_single_blocks_depth": 40,
            "rope_dim_list": [16, 56, 56],
            "hidden_size": 3072,
            "heads_num": 24,
            "mlp_width_ratio": 4,
            "guidance_embed": True,
        }
        with init_empty_weights():
            transformer = HYVideoDiffusionTransformer(
                in_channels=in_channels,
                out_channels=out_channels,
                attention_mode=attention_mode,
                main_device=device,
                offload_device=offload_device,
                **HUNYUAN_VIDEO_CONFIG,
                **factor_kwargs
            )
        transformer.eval()

        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        scheduler = FlowMatchDiscreteScheduler(
            shift=9.0,
            reverse=True,
            solver="euler",
        )
        pipe = HunyuanVideoPipeline(
            transformer=transformer,
            scheduler=scheduler,
            progress_bar_config=None,
            base_dtype=base_dtype
        )

        if not "torchao" in quantization:
            log.info("Using accelerate to load and assign model weights to device...")
            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast":
                dtype = torch.float8_e4m3fn
            else:
                dtype = base_dtype
            params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
            for name, param in transformer.named_parameters():
                dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            del sd
            gc.collect()
            mm.soft_empty_cache()

            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    if l["blocks"]:
                        lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])

                    #for k in lora_sd.keys():
                     #   print(k)

                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

            comfy.model_management.load_model_gpu(patcher)
            if load_device == "offload_device":
                patcher.model.diffusion_model.to(offload_device)

            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(patcher.model.diffusion_model, base_dtype, params_to_keep=params_to_keep)

            #compile
            if compile_args is not None:
                torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
                if compile_args["compile_single_blocks"]:
                    for i, block in enumerate(patcher.model.diffusion_model.single_blocks):
                        patcher.model.diffusion_model.single_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_double_blocks"]:
                    for i, block in enumerate(patcher.model.diffusion_model.double_blocks):
                        patcher.model.diffusion_model.double_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_txt_in"]:
                    patcher.model.diffusion_model.txt_in = torch.compile(patcher.model.diffusion_model.txt_in, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_vector_in"]:
                    patcher.model.diffusion_model.vector_in = torch.compile(patcher.model.diffusion_model.vector_in, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_final_layer"]:
                    patcher.model.diffusion_model.final_layer = torch.compile(patcher.model.diffusion_model.final_layer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
        elif "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                int4_weight_only
            )
            except:
                raise ImportError("torchao is not installed")

            # def filter_fn(module: nn.Module, fqn: str) -> bool:
            #     target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
            #     if any(sub in fqn for sub in target_submodules):
            #         return isinstance(module, nn.Linear)
            #     return False

            if "fp6" in quantization:
                quant_func = fpx_weight_only(3, 2)
            elif "int4" in quantization:
                quant_func = int4_weight_only()
            elif "int8" in quantization:
                quant_func = int8_weight_only()
            elif "fp8dq" in quantization:
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()

            log.info(f"Quantizing model with {quant_func}")
            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

            comfy.model_management.load_models_gpu([patcher])

            for i, block in enumerate(patcher.model.diffusion_model.single_blocks):
                log.info(f"Quantizing single_block {i}")
                for name, _ in block.named_parameters(prefix=f"single_blocks.{i}"):
                    #print(f"Parameter name: {name}")
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=patcher.model.diffusion_model_load_device, dtype=base_dtype, value=sd[name])
                if compile_args is not None:
                    patcher.model.diffusion_model.single_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                quantize_(block, quant_func)
                print(block)
                block.to(offload_device)
            for i, block in enumerate(patcher.model.diffusion_model.double_blocks):
                log.info(f"Quantizing double_block {i}")
                for name, _ in block.named_parameters(prefix=f"double_blocks.{i}"):
                    #print(f"Parameter name: {name}")
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=patcher.model.diffusion_model_load_device, dtype=base_dtype, value=sd[name])
                if compile_args is not None:
                    patcher.model.diffusion_model.double_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                quantize_(block, quant_func)
            for name, param in patcher.model.diffusion_model.named_parameters():
                if "single_blocks" not in name and "double_blocks" not in name:
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=patcher.model.diffusion_model_load_device, dtype=base_dtype, value=sd[name])

            manual_offloading = False # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")
            for name, param in patcher.model.diffusion_model.named_parameters():
                print(name, param.dtype)
                #param.data = param.data.to(self.vae_dtype).to(device)

            del sd
            mm.soft_empty_cache()

        patcher.model["pipe"] = pipe
        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = "disabled"
        patcher.model["block_swap_args"] = block_swap_args

        return (patcher,)


# A gentle text-encoder only micro-mod
class HunyuanNyanCLIP:
    transformer_deepcopy = None  # Class-level variable to store the deepcopy
    load_deepcopy_device = get_torch_device() 
    @classmethod
    def INPUT_TYPES(s):
        loader_inputs = HyVideoModelLoader.INPUT_TYPES()
        nyan_inputs = {
            "required": {
                "load_from_file": (["False", "True"], {"default": "True"}),
                "input_factor_clip": (["None", "Factor"],),
                "factor_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "input_factor_llm": (["None", "Factor"],),
                "factor_llm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            }
        }
        nyan_inputs["required"].update(loader_inputs["required"])
        nyan_inputs["optional"] = loader_inputs.get("optional", {})
        return nyan_inputs

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "modify"
    CATEGORY = "zer0int/HyNyan-Nyan"

    def scale_clip_influence(self, output, factor):
        """Scales the output of the CLIP embedding."""
        return factor * output

    def scale_llm_influence(self, output, factor):
        """Scales the output of the LLM embedding."""
        return factor * output

    def load_model(self, model, base_precision, load_device, quantization, **kwargs):
        """Uses HyVideoModelLoader's logic to load a model."""
        loader = HyVideoModelLoader()
        return loader.loadmodel(
            model=model,
            base_precision=base_precision,
            load_device=load_device,
            quantization=quantization,
            **kwargs
        )[0]  # Extract the patcher object

    def modify(self, **kwargs):
        load_from_file = kwargs.get("load_from_file", "True") == "True"
        device = get_torch_device()

        if load_from_file:
            # Load everything from file fresh every time
            log.info("Loading model from file...")
            model = self.load_model(
                model=kwargs["model"],
                base_precision=kwargs["base_precision"],
                load_device=kwargs["load_device"],
                quantization=kwargs["quantization"],
                compile_args=kwargs.get("compile_args"),
                attention_mode=kwargs.get("attention_mode", "sdpa"),
                block_swap_args=kwargs.get("block_swap_args"),
                lora=kwargs.get("lora"),
            )

            # Clear any previous deepcopy if loading from file
            if HunyuanNyanCLIP.transformer_deepcopy is not None:
                log.info("Clearing existing deepcopy as we are reloading from file...")
                del HunyuanNyanCLIP.transformer_deepcopy
                HunyuanNyanCLIP.transformer_deepcopy = None

        else:
            # Handle deepcopy logic
            if HunyuanNyanCLIP.transformer_deepcopy is None:
                # Load everything from file if deepcopy does not exist
                log.info("Deepcopy is None. Loading model from file...")
                model = self.load_model(
                    model=kwargs["model"],
                    base_precision=kwargs["base_precision"],
                    load_device=kwargs["load_device"],
                    quantization=kwargs["quantization"],
                    compile_args=kwargs.get("compile_args"),
                    attention_mode=kwargs.get("attention_mode", "sdpa"),
                    block_swap_args=kwargs.get("block_swap_args"),
                    lora=kwargs.get("lora"),
                )

                # Ensure the entire patcher is moved to CPU before deepcopy, else OOM
                log.info("Moving model to CPU for deepcopy...")
                patcher = model.model if hasattr(model, "model") else model
                patcher.to("cpu")
                HunyuanNyanCLIP.transformer_deepcopy = copy.deepcopy(patcher)
                log.info("Deepcopy created and stored in system memory.")
            else:
                # Restore model from deepcopy
                log.info("Restoring model from existing deepcopy in system memory...")
                patcher = copy.deepcopy(HunyuanNyanCLIP.transformer_deepcopy)

                # Rewrap the restored patcher into a HyVideoModel object with the required model_config
                log.info("Wrapping restored patcher into HyVideoModel...")
                model_config = HyVideoModelConfig(dtype=kwargs.get("base_precision", "bf16"))
                model = HyVideoModel(model_config=model_config, model_type=comfy.model_base.ModelType.FLOW, device=device)
                model.model = patcher

        patcher = model.model if hasattr(model, "model") else model
        transformer = patcher.diffusion_model if hasattr(patcher, "diffusion_model") else None
        if transformer is None:
            raise AttributeError("Could not locate the diffusion_model in the patcher structure.")

        transformer.to(device)

        # Scale CLIP influence
        if kwargs.get("input_factor_clip") == "Factor" and hasattr(transformer, "txt_in"):
            txt_in = transformer.txt_in
            if hasattr(txt_in, "c_embedder"):
                factor_clip = kwargs["factor_clip"]
                original_c_embedder_forward = txt_in.c_embedder.forward

                def scaled_c_embedder_forward(*args, **kwargs):
                    output = original_c_embedder_forward(*args, **kwargs)
                    return self.scale_clip_influence(output, factor_clip)

                txt_in.c_embedder.forward = scaled_c_embedder_forward

        # Scale LLM influence
        if kwargs.get("input_factor_llm") == "Factor" and hasattr(transformer, "txt_in"):
            txt_in = transformer.txt_in
            if hasattr(txt_in, "individual_token_refiner"):
                factor_llm = kwargs["factor_llm"]
                for i, block in enumerate(txt_in.individual_token_refiner.blocks):
                    original_block_forward = block.forward

                    def scaled_block_forward(*args, **kwargs):
                        output = original_block_forward(*args, **kwargs)
                        return self.scale_llm_influence(output, factor_llm)

                    block.forward = scaled_block_forward

        return (model,)


# The full CONFUSION for user and transformer
class HunyuanNyan:
    transformer_deepcopy = None  # Class-level variable to store the deepcopy
    load_deepcopy_device = get_torch_device() 
    @classmethod
    def INPUT_TYPES(s):
        loader_inputs = HyVideoModelLoader.INPUT_TYPES()
        nyan_inputs = {
            "required": {
                "load_from_file": (["False", "True"], {"default": "True"}),
                "input_factor_clip": (["None", "Factor"],),
                "factor_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "input_factor_llm": (["None", "Factor"],),
                "factor_llm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "shuffle_double_attn_img": (["None", "Shuffle", "Skip"],),
                "double_attn_img": ("STRING", {"default": ""}),
                "shuffle_double_attn_txt": (["None", "Shuffle", "Skip"],),
                "double_attn_txt": ("STRING", {"default": ""}),
                "shuffle_double_mlp_img": (["None", "Shuffle", "Skip"],),
                "double_mlp_img": ("STRING", {"default": ""}),
                "shuffle_double_mlp_txt": (["None", "Shuffle", "Skip"],),
                "double_mlp_txt": ("STRING", {"default": ""}),
                "shuffle_single_attn_img": (["None", "Shuffle", "Skip"],),
                "single_attn_img": ("STRING", {"default": ""}),
                "shuffle_single_attn_txt": (["None", "Shuffle", "Skip"],),
                "single_attn_txt": ("STRING", {"default": ""}),
                "shuffle_single_mlp_img": (["None", "Shuffle", "Skip"],),
                "single_mlp_img": ("STRING", {"default": ""}),
                "shuffle_single_mlp_txt": (["None", "Shuffle", "Skip"],),
                "single_mlp_txt": ("STRING", {"default": ""}),
            }
        }
        nyan_inputs["required"].update(loader_inputs["required"])
        nyan_inputs["optional"] = loader_inputs.get("optional", {})
        return nyan_inputs

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "modify"
    CATEGORY = "zer0int/HyNyan-Nyan"

    def parse_layers(self, layer_str):
        """Parses a comma-separated string of layer indices into a sorted list of integers."""
        if not layer_str.strip():
            return []
        layers = [int(x) for x in re.split(r"[,\s]+", layer_str) if x.isdigit()]
        return sorted(layers)

    def process_layers(self, layer_indices, action, block_list, component_type):
        """Processes specific layers based on the action (None, Shuffle, Skip)."""
        if action == "None":
            return
        if action == "Skip":
            for idx in range(len(layer_indices) - 1):
                current_idx, target_idx = layer_indices[idx], layer_indices[idx + 1]
                current_block = getattr(block_list[current_idx], component_type)
                target_block = getattr(block_list[target_idx], component_type)
                current_block.forward = lambda *args, **kwargs: target_block.forward(*args, **kwargs)
        elif action == "Shuffle":
            n = len(layer_indices)
            for i in range(n // 2):
                idx1, idx2 = layer_indices[i], layer_indices[n - i - 1]
                block1, block2 = getattr(block_list[idx1], component_type), getattr(block_list[idx2], component_type)
                setattr(block_list[idx1], component_type, block2)
                setattr(block_list[idx2], component_type, block1)

    def scale_clip_influence(self, output, factor):
        """Scales the output of the CLIP embedding."""
        return factor * output

    def scale_llm_influence(self, output, factor):
        """Scales the output of the LLM embedding."""
        return factor * output

    def load_model(self, model, base_precision, load_device, quantization, **kwargs):
        """Uses HyVideoModelLoader's logic to load a model."""
        loader = HyVideoModelLoader()
        return loader.loadmodel(
            model=model,
            base_precision=base_precision,
            load_device=load_device,
            quantization=quantization,
            **kwargs
        )[0]  # Extract the patcher object

    def modify(self, **kwargs):
        load_from_file = kwargs.get("load_from_file", "True") == "True"
        device = get_torch_device()

        if load_from_file:
            # Load everything from file fresh every time
            log.info("Loading model from file...")
            model = self.load_model(
                model=kwargs["model"],
                base_precision=kwargs["base_precision"],
                load_device=kwargs["load_device"],
                quantization=kwargs["quantization"],
                compile_args=kwargs.get("compile_args"),
                attention_mode=kwargs.get("attention_mode", "sdpa"),
                block_swap_args=kwargs.get("block_swap_args"),
                lora=kwargs.get("lora"),
            )

            # Clear any previous deepcopy if loading from file
            if HunyuanNyan.transformer_deepcopy is not None:
                log.info("Clearing existing deepcopy as we are reloading from file...")
                del HunyuanNyan.transformer_deepcopy
                HunyuanNyan.transformer_deepcopy = None

        else:
            # Handle deepcopy logic
            if HunyuanNyan.transformer_deepcopy is None:
                # Load everything from file if deepcopy does not exist
                log.info("Deepcopy is None. Loading model from file...")
                model = self.load_model(
                    model=kwargs["model"],
                    base_precision=kwargs["base_precision"],
                    load_device=kwargs["load_device"],
                    quantization=kwargs["quantization"],
                    compile_args=kwargs.get("compile_args"),
                    attention_mode=kwargs.get("attention_mode", "sdpa"),
                    block_swap_args=kwargs.get("block_swap_args"),
                    lora=kwargs.get("lora"),
                )

                # Ensure the entire patcher is moved to CPU before deepcopy, else OOM
                log.info("Moving model to CPU for deepcopy...")
                patcher = model.model if hasattr(model, "model") else model
                patcher.to("cpu")
                HunyuanNyan.transformer_deepcopy = copy.deepcopy(patcher)
                log.info("Deepcopy created and stored in system memory.")
            else:
                # Restore model from deepcopy
                log.info("Restoring model from existing deepcopy in system memory...")
                patcher = copy.deepcopy(HunyuanNyan.transformer_deepcopy)

                # Rewrap the restored patcher into a HyVideoModel object with the required model_config
                log.info("Wrapping restored patcher into HyVideoModel...")
                model_config = HyVideoModelConfig(dtype=kwargs.get("base_precision", "bf16"))
                model = HyVideoModel(model_config=model_config, model_type=comfy.model_base.ModelType.FLOW, device=device)
                model.model = patcher

        patcher = model.model if hasattr(model, "model") else model
        transformer = patcher.diffusion_model if hasattr(patcher, "diffusion_model") else None
        if transformer is None:
            raise AttributeError("Could not locate the diffusion_model in the patcher structure.")

        transformer.to(device)

        # NYAN CONFUSION Layer Shuffle
        double_attn_img_layers = self.parse_layers(kwargs.get("double_attn_img", ""))
        double_attn_txt_layers = self.parse_layers(kwargs.get("double_attn_txt", ""))
        double_mlp_img_layers = self.parse_layers(kwargs.get("double_mlp_img", ""))
        double_mlp_txt_layers = self.parse_layers(kwargs.get("double_mlp_txt", ""))
        single_attn_img_layers = self.parse_layers(kwargs.get("single_attn_img", ""))
        single_attn_txt_layers = self.parse_layers(kwargs.get("single_attn_txt", ""))
        single_mlp_img_layers = self.parse_layers(kwargs.get("single_mlp_img", ""))
        single_mlp_txt_layers = self.parse_layers(kwargs.get("single_mlp_txt", ""))

        # Process double_blocks
        if hasattr(transformer, "double_blocks"):
            double_blocks = transformer.double_blocks
            self.process_layers(double_attn_img_layers, kwargs["shuffle_double_attn_img"], double_blocks, "img_attn_proj")
            self.process_layers(double_attn_txt_layers, kwargs["shuffle_double_attn_txt"], double_blocks, "txt_attn_proj")
            self.process_layers(double_mlp_img_layers, kwargs["shuffle_double_mlp_img"], double_blocks, "img_mlp")
            self.process_layers(double_mlp_txt_layers, kwargs["shuffle_double_mlp_txt"], double_blocks, "txt_mlp")

        # Process single_blocks
        if hasattr(transformer, "single_blocks"):
            single_blocks = transformer.single_blocks
            self.process_layers(single_attn_img_layers, kwargs["shuffle_single_attn_img"], single_blocks, "modulation")
            self.process_layers(single_attn_txt_layers, kwargs["shuffle_single_attn_txt"], single_blocks, "modulation")
            self.process_layers(single_mlp_img_layers, kwargs["shuffle_single_mlp_img"], single_blocks, "linear1")
            self.process_layers(single_mlp_txt_layers, kwargs["shuffle_single_mlp_txt"], single_blocks, "linear1")

        # Scale CLIP influence
        if kwargs.get("input_factor_clip") == "Factor" and hasattr(transformer, "txt_in"):
            txt_in = transformer.txt_in
            if hasattr(txt_in, "c_embedder"):
                factor_clip = kwargs["factor_clip"]
                original_c_embedder_forward = txt_in.c_embedder.forward

                def scaled_c_embedder_forward(*args, **kwargs):
                    output = original_c_embedder_forward(*args, **kwargs)
                    return self.scale_clip_influence(output, factor_clip)

                txt_in.c_embedder.forward = scaled_c_embedder_forward

        # Scale LLM influence
        if kwargs.get("input_factor_llm") == "Factor" and hasattr(transformer, "txt_in"):
            txt_in = transformer.txt_in
            if hasattr(txt_in, "individual_token_refiner"):
                factor_llm = kwargs["factor_llm"]
                for i, block in enumerate(txt_in.individual_token_refiner.blocks):
                    original_block_forward = block.forward

                    def scaled_block_forward(*args, **kwargs):
                        output = original_block_forward(*args, **kwargs)
                        return self.scale_llm_influence(output, factor_llm)

                    block.forward = scaled_block_forward

        return (model,)
