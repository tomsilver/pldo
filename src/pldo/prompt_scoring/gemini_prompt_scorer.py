"""Base class for prompt generators."""

import logging
from string import Template

from google import genai
from google.genai import types
from PIL import Image
from prpl_perception_utils.structs import RGBImage

from pldo.prompt_scoring.base_prompt_scorer import PromptScorer

_PROMPT = Template(
    """
    You will be shown an image and a text prompt describing an object.

    Decide whether the object described is present in the image — even if the
    viewpoint, lighting, or occlusion may differ.

    Answer strictly with one of the following:
    - "Yes" if the object appears to be present.
    - "No" if the object is not present or clearly mismatches the description.

    Prompt: "$prompt"
    """
)


class GeminiPromptScorer(PromptScorer):
    """Score several unique candidate prompts based on detection of an object."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        thumbnail_size: int = 1024,
        min_mask_value: int = 100,
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

    def score(self, rgb: RGBImage, prompt: str, negative: bool) -> int:
        """Returns 1 if model correctly detects presence/absence of object, else 0."""

        full_prompt = _PROMPT.substitute(prompt=prompt)

        im = Image.fromarray(rgb)
        im.thumbnail(
            (self._thumbnail_size, self._thumbnail_size),
            Image.Resampling.LANCZOS,
        )

        prompt_contents: list[str | Image.Image] = [full_prompt, im]

        logging.info("Querying Gemini for binary object detection")

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt_contents,  # type: ignore
            config=self._config,
        )

        assert response.text is not None
        answer = response.text.strip().lower()

        # Normalize model output
        if "yes" in answer:
            detected = True
        elif "no" in answer:
            detected = False
        else:
            logging.warning(f"Unexpected model output: {response.text}")
            return 0

        # Determine correctness
        expected = not negative
        return int(detected == expected)
