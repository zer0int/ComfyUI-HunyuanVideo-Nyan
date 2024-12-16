from . import hynyan as hynyan

NODE_CLASS_MAPPINGS = {
    "HunyuanNyan": hynyan.HunyuanNyan,
    "HunyuanNyanCLIP": hynyan.HunyuanNyanCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanNyan": "Hunyuan Nyan-Shuffle",
    "HunyuanNyanCLIP": "Hunyuan Nyan-CLIP",
}