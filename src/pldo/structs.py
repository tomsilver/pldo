"""Data structures."""

from dataclasses import dataclass

from prpl_perception_utils.structs import RGBImage


@dataclass(frozen=True)
class PromptSearchDataset:
    """Training and validation data for prompt search."""

    train_rgbs: list[RGBImage]  # used for prompt generation
    val_positive_rgbs: list[RGBImage]  # positive images of the object
    val_negative_rgbs: list[RGBImage]  # negative images of the object
