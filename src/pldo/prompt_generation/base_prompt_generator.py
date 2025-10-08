"""Base class for prompt generators."""

import abc

from prpl_perception_utils.structs import RGBImage


class PromptGenerator(abc.ABC):
    """Base class for image-to-text prompt generators."""

    @abc.abstractmethod
    def generate(self, rgbs: list[RGBImage], num_samples: int) -> list[str]:
        """Generate several unique candidate prompts for a given object."""
