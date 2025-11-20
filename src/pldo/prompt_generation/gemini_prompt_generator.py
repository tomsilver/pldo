"""Gemini version of an object-centric prompt generator."""

import re
from string import Template

from prpl_llm_utils.cache import (
    PretrainedLargeModelCache,
)
from prpl_llm_utils.reprompting import (
    create_reprompt_from_error_message,
)
from prpl_llm_utils.structs import Query, Response
from prpl_perception_utils.structs import (
    RGBImage,
)

from pldo.gemini_client.base_gemini_client import GeminiClient
from pldo.prompt_generation.base_prompt_generator import PromptGenerator

_PROMPT = Template(
    r"""
    You are shown multiple views of the same object named:
    $initial_prompt

    1. For each image, briefly describe the object — including color, shape,
        and distinctive traits.
    2. Output these attributes separately:
    Color: ...
    Shape: ...
    Distinctive traits: ...

    3. Based on these attributes, generate $num_samples distinct, diverse, and
        concise combined description(s) that generalize(s) across all views, occlusion,
        and slight lighting changes while varying in scope, specificity, and context.
    Combined description 1: "A ... with ..."
    etc.

    Output a final JSON list with format "json\s*(\[\s*.*?\s*\])\s*",
    with all combined descriptions.

    These will be used as candidate prompt(s) for an algorithm that mixes and
    selects the best descriptions for a robotics-focused object detection task.
    Focus more on structure than function, and be specific about unique structural
    details that help make the object more easily identifiable.
    """
)


class GeminiPromptGenerator(PromptGenerator, GeminiClient):
    """Gemini-based image-to-text prompt generators."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        thumbnail_size: int = 1024,
        min_mask_value: int = 100,
        cache: PretrainedLargeModelCache | None = None,
        initial_prompt: str = "",
        use_cache_only: bool = False,
        thinking_budget: int = 0,
    ) -> None:

        GeminiClient.__init__(
            self,
            model_name,
            thumbnail_size,
            min_mask_value,
            cache,
            use_cache_only,
            thinking_budget,
        )

        # Initial prompt
        self._initial_prompt = initial_prompt

        # Number of samples
        self._num_samples = 0

    @property
    def initial_prompt(self):
        """Initial prompt getter method."""
        return self._initial_prompt

    def reprompt_and_parse(self, query: Query, response: Response) -> Query | None:
        """Check a response, and force a reprompt if the format is incorrect."""

        if not response.text:
            return create_reprompt_from_error_message(
                query, response, "Response was empty."
            )

        json_match = re.search(r"json\s*(\[\s*.*?\s*\])\s*", response.text, re.DOTALL)

        if not json_match:
            return create_reprompt_from_error_message(
                query, response, "Could not find JSON block."
            )

        if not json_match.group(1):
            return create_reprompt_from_error_message(
                query, response, "Could not find the first group of the JSON block."
            )

        prompt_list = eval(json_match.group(1))
        if len(prompt_list) != self._num_samples:
            return create_reprompt_from_error_message(
                query,
                response,
                "Wrong number of prompts returned. "
                f"Expected {self._num_samples} prompts.",
            )

        # Valid response format (no reprompt)
        response.metadata["parsed"] = prompt_list
        return None

    def generate(self, rgbs: list[RGBImage], num_samples: int) -> list[str]:
        """Generate several unique candidate prompts for a given object."""

        # Build the generator prompt
        self._num_samples = num_samples
        prompt = _PROMPT.substitute(
            initial_prompt=self._initial_prompt,
            num_samples=num_samples,
        )

        # Query and parse, return the result
        return self.query_and_parse(prompt, rgbs)
