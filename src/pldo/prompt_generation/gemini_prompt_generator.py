"""Base class for prompt generators."""

import logging
import re
from string import Template

from google import genai
from google.genai import types
from PIL import Image
from prpl_perception_utils.structs import RGBImage

from pldo.prompt_generation.base_prompt_generator import PromptGenerator

_PROMPT = Template(
    """
    You are shown multiple views of the same object named:
    $initial_prompt

    1. For each image, briefly describe the object — including color, shape,
        and distinctive traits.
    2. Output these attributes separately:
    Color: ...
    Shape: ...
    Distinctive traits: ...

    3. Based on these attributes, generate $num_samples distinct, diverse, and
        concise combined descriptions that generalize across all views, occlusion,
        and lighting changes while varying in scope, specificity, and context.
    Combined description 1: "A ... with ..."
    etc.

    Output a final JSON list with all combined descriptions.

    These will be used as candidate prompts for an algorithm that mixes and
    selects the best descriptions for a robotics-focused object detection task.
    Focus more on structure than function, and be specific about unique structural
    details that help make the object more easily identifiable.
    """
)


class GeminiPromptGenerator(PromptGenerator):
    """Gemini-based image-to-text prompt generators."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        thumbnail_size: int = 1024,
        min_mask_value: int = 100,
        initial_prompt: str = "",
        test: bool = False,
    ) -> None:
        self._model = model
        self._thumbnail_size = thumbnail_size
        if not test:
            self._client = genai.Client()
            self._config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            self._min_mask_value = min_mask_value
        self._initial_prompt = initial_prompt

    @property
    def initial_prompt(self):
        """Initial prompt getter method."""
        return self._initial_prompt

    def generate(self, rgbs: list[RGBImage], num_samples: int) -> list[str]:
        """Generate several unique candidate prompts for a given object."""

        # Build the generator prompt
        generator_prompt = _PROMPT.substitute(
            initial_prompt=self._initial_prompt, num_samples=num_samples
        )

        # Convert each RGB image to a resized PIL Image
        images: list[Image.Image] = []
        for rgb in rgbs:
            im = Image.fromarray(rgb)
            im.thumbnail(
                (self._thumbnail_size, self._thumbnail_size), Image.Resampling.LANCZOS
            )
            images.append(im)

        # Combine the text prompt and image list
        prompt_contents: list[str | Image.Image] = [generator_prompt] + images

        logging.info(f"Sending {len(images)} image(s) to Gemini")

        # Query the Gemini model
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt_contents,  # type: ignore
            config=self._config,
        )

        logging.info("Received response from Gemini")

        assert response.text is not None

        json_match = re.search(r"json\s*(\[\s*.*?\s*\])\s*", response.text, re.DOTALL)

        if not json_match:
            raise ValueError("Could not find JSON block in the input")

        prompt_list = eval(json_match.group(1))

        return prompt_list
