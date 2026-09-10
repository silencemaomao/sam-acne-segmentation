from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class SAMEncoderSegmenter(nn.Module):
    """Prompt-free semantic segmenter built on SAM's pretrained image encoder."""

    def __init__(
        self,
        pretrained_name: str = "facebook/sam-vit-base",
        decoder_channels: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import SamModel

        sam = SamModel.from_pretrained(pretrained_name, local_files_only=local_files_only)
        self.encoder = sam.vision_encoder
        self.expected_image_size = int(self.encoder.config.image_size)
        encoder_channels = int(self.encoder.config.output_channels)
        self.encoder_frozen = bool(freeze_encoder)
        if self.encoder_frozen:
            self.encoder.requires_grad_(False)
        elif gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 4, num_classes, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        expected = self.expected_image_size
        if (height, width) != (expected, expected):
            raise ValueError(
                f"SAM encoder expects {(expected, expected)} because of absolute positional embeddings, "
                f"but received {(height, width)}."
            )
        if self.encoder_frozen:
            with torch.no_grad():
                features = self.encoder(pixel_values=images, return_dict=True).last_hidden_state
        else:
            features = self.encoder(pixel_values=images, return_dict=True).last_hidden_state
        logits = self.decoder(features)
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)


class TinySegmenter(nn.Module):
    def __init__(self, channels: int = 16, num_classes: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    model_type = model_config.pop("type")
    class_count = len(config["data"]["classes"])
    configured_count = model_config.pop("num_classes", None)
    if configured_count is not None and int(configured_count) != class_count:
        raise ValueError(
            f"model.num_classes={configured_count} does not match "
            f"len(data.classes)={class_count}. Use null for automatic inference."
        )
    model_config["num_classes"] = class_count
    if model_type == "sam_encoder_segmenter":
        return SAMEncoderSegmenter(**model_config)
    if model_type == "tiny_segmenter":
        return TinySegmenter(**model_config)
    raise ValueError(f"Unknown model.type: {model_type}")
