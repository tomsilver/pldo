"""Gemini version of a prompt mutator."""

import json
import re
from string import Template

from prpl_llm_utils.cache import (
    PretrainedLargeModelCache,
)
from prpl_llm_utils.reprompting import (
    create_reprompt_from_error_message,
)
from prpl_llm_utils.structs import Query, Response

from pldo.gemini_client.base_gemini_client import GeminiClient
from pldo.prompt_mutation.base_prompt_mutator import PromptMutator

_PROMPT = Template(
    r"""
    You are optimizing object description prompts for a visual recognition system.
    Below are several high-performing prompts that describe the same object class,
    but each uses slightly different wording and emphasis.

    High-performing prompts:
    $winner_prompts

    1. Carefully read all the prompts. Identify their shared structure and key
       descriptive elements (e.g., color, shape, distinctive traits, composition).
    2. Identify what varies across them — e.g., phrasing, specificity, context,
       or focus.
    3. Generate ONE new prompt that combines the strengths of these examples:
       - Preserve the core meaning and discriminative features.
       - Slightly vary the style, specificity, or scope to increase diversity.
       - Stay concise and factual, focusing on structure and distinguishing details.
       - Avoid redundancy or overly generic phrasing.

    The result should be a single sentence suitable as an object detection prompt.

    Output format (strictly):
    "json\s*(\[\s*\"(.*?)\"\s*\])\s*"

    Example output:
    json ["A compact red toolbox with metallic hinges and a rectangular lid"]

    Remember: the goal is **mutation**, not rewriting — keep it close in meaning
    but sufficiently distinct to test new phrasing hypotheses.
    """
)


class GeminiPromptMutator(PromptMutator, GeminiClient):
    """Gemini-based image-to-text prompt generators."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
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
        """Validate the model output for the mutator and reprompt if malformed."""

        if not response.text:
            return create_reprompt_from_error_message(
                query, response, "Response was empty."
            )

        text = response.text.strip()

        # Match the expected pattern: json ["..."]
        match = re.search(r'json\s*(\[\s*".*?"\s*\])', text, re.IGNORECASE | re.DOTALL)
        if not match:
            return create_reprompt_from_error_message(
                query,
                response,
                f"Could not find valid JSON list in response: {response.text}",
            )

        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return create_reprompt_from_error_message(
                query,
                response,
                f"JSON decoding error: {e}",
            )

        # Must be a non-empty list of strings
        if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], str):
            return create_reprompt_from_error_message(
                query,
                response,
                f"Unexpected JSON format: expected list of strings, got {parsed!r}",
            )

        # Valid response format (no reprompt)
        response.metadata["parsed"] = parsed[0].strip()
        return None

    def mutate(self, winners: list[str]) -> str:
        """Returns 1 if model correctly detects presence/absence of object, else 0."""

        # Build the generator prompt
        prompt = _PROMPT.substitute(winner_prompts="\n".join(f"> {p}" for p in winners))

        # Query and parse, return the result
        return self.query_and_parse(prompt, [], temperature=1.0)
