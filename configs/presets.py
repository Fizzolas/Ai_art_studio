"""
Training presets optimized for RTX 4070 Laptop GPU (8GB VRAM).
Each preset configures the entire training pipeline for a specific
base model + VRAM constraint combination.
"""

PRESETS = {
    "sd15_lora_8gb": {
        "name": "SD 1.5 LoRA (8GB Safe)",
        "description": "Stable Diffusion 1.5 LoRA training. Fits comfortably in 8GB VRAM.",
        "config": {
            "base_model": "runwayml/stable-diffusion-v1-5",
            "model_type": "sd15",
            "training_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 16,
            "learning_rate": 0.0001,
            "unet_lr": 0.0001,
            "text_encoder_lr": 0.00005,
            "batch_size": 1,
            "max_train_steps": 2000,
            "resolution": 512,
            "optimizer": "AdamW8bit",
            "lr_scheduler": "cosine",
            "mixed_precision": "fp16",
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "fp8_base": False,
            "noise_offset": 0.05,
            "enable_bucket": True,
        },
    },

    "sdxl_lora_8gb": {
        "name": "SDXL LoRA (8GB Tight)",
        "description": "SDXL LoRA training with aggressive memory optimizations. Uses fused backward pass.",
        "config": {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_type": "sdxl",
            "training_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 16,
            "learning_rate": 0.0003,
            "unet_lr": 0.0004,
            "text_encoder_lr": 0.0003,
            "batch_size": 1,
            "max_train_steps": 2000,
            "resolution": 512,
            "optimizer": "Adafactor",
            "optimizer_args": "scale_parameter=False relative_step=False warmup_init=False",
            "lr_scheduler": "constant",
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "fp8_base": True,
            "full_bf16": True,
            "noise_offset": 0.05,
            "enable_bucket": True,
            "max_bucket_reso": 1024,
        },
    },

    "flux_lora_8gb": {
        "name": "FLUX LoRA (8GB Split Mode)",
        "description": "FLUX.1 LoRA training with split_mode for 8GB VRAM. 512x512 max resolution.",
        "config": {
            "base_model": "black-forest-labs/FLUX.1-dev",
            "model_type": "flux",
            "training_type": "lora",
            "lora_rank": 32,
            "lora_alpha": 32,
            "network_module": "networks.lora_flux",
            "unet_lr": 0.0004,
            "batch_size": 1,
            "max_train_steps": 800,
            "resolution": 512,
            "optimizer": "AdamW8bit",
            "lr_scheduler": "constant",
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs": True,
            "cache_text_encoder_outputs_to_disk": True,
            "gradient_checkpointing": True,
            "fp8_base": True,
            "full_bf16": True,
            "split_mode": True,
            "noise_offset": 0.05,
            "enable_bucket": True,
        },
    },

    "sdxl_character_lora": {
        "name": "SDXL Character LoRA (Recommended)",
        "description": "Optimized for character training with diverse datasets. Best for your use case.",
        "config": {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_type": "sdxl",
            "training_type": "lora",
            "lora_rank": 64,
            "lora_alpha": 32,
            "learning_rate": 0.0003,
            "unet_lr": 0.0004,
            "text_encoder_lr": 0.00015,
            "batch_size": 1,
            "max_train_steps": 3000,
            "resolution": 512,
            "optimizer": "Adafactor",
            "optimizer_args": "scale_parameter=False relative_step=False warmup_init=False",
            "lr_scheduler": "constant",
            "mixed_precision": "bf16",
            "save_precision": "bf16",
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "fp8_base": True,
            "full_bf16": True,
            "noise_offset": 0.1,
            "enable_bucket": True,
            "flip_aug": False,  # Don't flip characters - asymmetric features matter
            "min_snr_gamma": 5.0,
            "save_every_n_steps": 500,
        },
    },
}


def get_preset(name: str) -> dict:
    return PRESETS.get(name, {})


def list_presets() -> list:
    return [
        {"key": k, "name": v["name"], "description": v["description"]}
        for k, v in PRESETS.items()
    ]
