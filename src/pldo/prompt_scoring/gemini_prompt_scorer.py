"""Gemini version of a prompt scorer."""

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
from pldo.prompt_scoring.base_prompt_scorer import PromptScorer

_PROMPT = Template(
    """
    You will be shown an image and a text prompt describing an object.

    Decide whether the object described is present in the image — even if the
    viewpoint, lighting, or occlusion may differ. The overall color hue, shape,
    and distinctive traits must match.

    Answer strictly with one of the following:
    - "Yes" if the object appears to be present.
    - "No" if the object is not present or clearly mismatches the description.

    Prompt: "$gen_prompt"
    """
)


class GeminiPromptScorer(PromptScorer, GeminiClient):
    """Gemini-based image-to-text prompt generators."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        thumbnail_size: int = 1024,
        min_mask_value: int = 100,
        cache: PretrainedLargeModelCache | None = None,
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

    def reprompt_and_parse(self, query: Query, response: Response) -> Query | None:
        """Check a response, and force a reprompt if the format is incorrect."""

        if not response.text:
            return create_reprompt_from_error_message(
                query, response, "Response was empty."
            )

        answer = response.text.strip().lower()

        # Normalize model output
        if "yes" in answer and "no" not in answer:
            detected = True
        elif "no" in answer and "yes" not in answer:
            detected = False
        else:
            return create_reprompt_from_error_message(
                query, response, f"Unexpected model output: {response.text}"
            )

        # Valid response format (no reprompt)
        response.metadata["parsed"] = detected
        return None

    def score(self, rgb: RGBImage, gen_prompt: str, negative: bool) -> int:
        """Returns 1 if model correctly detects presence/absence of object, else 0."""

        # Build the generator prompt
        prompt = _PROMPT.substitute(
            gen_prompt=gen_prompt,
        )

        # Query and parse, return the result
        detected = self.query_and_parse(prompt, [rgb])
        expected = not negative
        return int(detected == expected)
