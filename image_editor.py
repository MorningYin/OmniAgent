"""Image editor engine wrapping Qwen-Image-Edit-2511 with Lightning LoRA."""

import logging
import math
import os

import torch
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)

# Lightning scheduler 配置 (与参考脚本对齐)
LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class ImageEditorEngine:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None

    def load(self):
        if self.pipeline is not None:
            return
        logger.info("Loading Image Editor from %s", self.config.editor_path)

        from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
        from diffusers.models import QwenImageTransformer2DModel

        # 单独加载 transformer 再传给 pipeline（与参考脚本一致）
        transformer = QwenImageTransformer2DModel.from_pretrained(
            self.config.editor_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            LIGHTNING_SCHEDULER_CONFIG,
        )

        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.config.editor_path,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        )

        # 加载 Lightning LoRA (直接传完整路径)
        lora_path = self.config.lightning_lora_path
        logger.info("Loading Lightning LoRA from %s", lora_path)
        self.pipeline.load_lora_weights(lora_path)

        # 上 GPU + 开启 VAE tiling
        try:
            self.pipeline.to(self.config.editor_device)
            self.pipeline.vae.enable_tiling()
            logger.info("Image Editor loaded on %s", self.config.editor_device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM on %s, falling back to CPU offload", self.config.editor_device)
            gpu_id = int(self.config.editor_device.split(":")[-1])
            self.pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
            self.pipeline.vae.enable_tiling()
            logger.info("Image Editor loaded (CPU offload mode, gpu_id=%d)", gpu_id)

    def edit(self, image: Image.Image, instruction: str) -> Image.Image:
        """Apply an edit instruction to the image and return the edited result."""
        if self.pipeline is None:
            raise RuntimeError("Image editor not loaded. Call load() first.")

        logger.info("Editing image with instruction: %s", instruction)

        generator = torch.Generator(device=self.config.editor_device).manual_seed(42)
        output = self.pipeline(
            image=image,
            prompt=instruction,
            negative_prompt=self.config.editor_negative_prompt,
            true_cfg_scale=self.config.editor_true_cfg_scale,
            guidance_scale=self.config.editor_guidance_scale,
            num_inference_steps=self.config.editor_num_steps,
            generator=generator,
        )

        edited_image = output.images[0]
        logger.info("Image editing completed")
        return edited_image
