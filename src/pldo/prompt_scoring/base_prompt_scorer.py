"""Base class for prompt scorers."""

import abc

from prpl_perception_utils.structs import RGBImage


class PromptScorer(abc.ABC):
    """Base class for image-to-text prompt scorers."""

    @abc.abstractmethod
    def score(self, rgb: RGBImage, gen_prompt: str, negative: bool) -> int:
        """Score several unique candidate prompts based on detection of an object."""
