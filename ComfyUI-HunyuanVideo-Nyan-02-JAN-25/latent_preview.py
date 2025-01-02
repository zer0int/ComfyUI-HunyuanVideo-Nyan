import torch
from PIL import Image

from comfy.cli_args import args, LatentPreviewMethod
import comfy.model_management
import comfy.utils

MAX_PREVIEW_RESOLUTION = args.preview_size

def preview_to_image(latent_image):
        latents_ubyte = (((latent_image + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            ).to(device="cpu", dtype=torch.uint8, non_blocking=comfy.model_management.device_supports_non_blocking(latent_image.device))

        return Image.fromarray(latents_ubyte.numpy())

class LatentPreviewer:
    def decode_latent_to_preview(self, x0):
        pass

    def decode_latent_to_preview_image(self, preview_format, x0):
        preview_image = self.decode_latent_to_preview(x0)
        return ("GIF", preview_image, MAX_PREVIEW_RESOLUTION)

class Latent2RGBPreviewer(LatentPreviewer):
    def __init__(self):
        latent_rgb_factors = [[-0.41, -0.25, -0.26], 
                              [-0.26, -0.49, -0.24], 
                              [-0.37, -0.54, -0.3], 
                              [-0.04, -0.29, -0.29], 
                              [-0.52, -0.59, -0.39], 
                              [-0.56, -0.6, -0.02], 
                              [-0.53, -0.06, -0.48], 
                              [-0.51, -0.28, -0.18], 
                              [-0.59, -0.1, -0.33], 
                              [-0.56, -0.54, -0.41], 
                              [-0.61, -0.19, -0.5], 
                              [-0.05, -0.25, -0.17], 
                              [-0.23, -0.04, -0.22], 
                              [-0.51, -0.56, -0.43], 
                              [-0.13, -0.4, -0.05], 
                              [-0.01, -0.01, -0.48]]
        self.latent_rgb_factors = torch.tensor(latent_rgb_factors, device="cpu").transpose(0, 1)
        self.latent_rgb_factors_bias = torch.tensor([0.138, 0.025, -0.299], device="cpu")

    def decode_latent_to_preview(self, x0):
        self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
        if self.latent_rgb_factors_bias is not None:
            self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)

        latent_image = torch.nn.functional.linear(x0[0].permute(1, 2, 0), self.latent_rgb_factors,
                                                    bias=self.latent_rgb_factors_bias)
        return preview_to_image(latent_image)


def get_previewer():
    previewer = None
    method = args.preview_method
    if method != LatentPreviewMethod.NoPreviews:
        # TODO previewer method

        if method == LatentPreviewMethod.Auto:
            method = LatentPreviewMethod.Latent2RGB

        if previewer is None:
            previewer = Latent2RGBPreviewer()
    return previewer

def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = get_previewer()

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

