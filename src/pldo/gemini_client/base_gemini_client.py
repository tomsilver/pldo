"""Base class for Gemini clients."""

import abc
import logging
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from prpl_llm_utils.cache import (
    PretrainedLargeModelCache,
    SQLite3PretrainedLargeModelCache,
)
from prpl_llm_utils.models import GeminiModel, PretrainedLargeModel
from prpl_llm_utils.reprompting import (
    FunctionalRepromptCheck,
    query_with_reprompts,
)
from prpl_llm_utils.structs import Query, Response
from prpl_perception_utils.structs import RGBImage


class DummyGeminiModel(PretrainedLargeModel):
    """API-free Gemini interface, for testing."""

    def __init__(
        self,
        model_name: str,
        cache: PretrainedLargeModelCache,
        use_cache_only: bool = False,
    ) -> None:
        self._model_name = model_name
        super().__init__(cache, use_cache_only)

    def get_id(self) -> str:
        return self._model_name

    def _run_query(self, query: Query) -> Response:
        raise NotImplementedError(
            "Dummy Gemini model cannot process queries without a cache."
        )


class GeminiClient(abc.ABC):
    """Gemini client module, capable of querying, caching, reprompting, and parsing."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        thumbnail_size: int = 1024,
        min_mask_value: int = 100,
        cache: PretrainedLargeModelCache | None = None,
        use_cache_only: bool = False,
        thinking_budget: int = 0,
    ) -> None:

        if cache is None:
            logging.info("Cache does not exist, a default one will be created")
            cache_path = Path(tempfile.gettempdir()) / "generation.db"
            cache = SQLite3PretrainedLargeModelCache(cache_path)

        self._gemini_model: PretrainedLargeModel

        if use_cache_only:
            self._gemini_model = DummyGeminiModel(
                model_name=model_name,
                cache=cache,
                use_cache_only=use_cache_only,
            )
        else:
            self._gemini_model = GeminiModel(
                model_name=model_name,
                cache=cache,
                use_cache_only=use_cache_only,
                thinking_budget=thinking_budget,
            )
        self._thumbnail_size = thumbnail_size
        self._min_mask_value = min_mask_value

    @abc.abstractmethod
    def reprompt_and_parse(self, query: Query, response: Response) -> Query | None:
        """
        Check a response, and force a reprompt if the format is incorrect.
        If the format is correct, store a parsed output in response.metadata["parsed"].
        """

    def query_and_parse(
        self,
        prompt: str,
        rgbs: list[RGBImage],
        temperature: float = 0.0,
    ) -> Any:
        """Query with reprompts, and return the parsed result (if it exists)."""

        # Convert each RGB image to a resized PIL Image
        images: list[Image.Image] = []
        for rgb in rgbs:
            im = Image.fromarray(rgb)
            im.thumbnail(
                (self._thumbnail_size, self._thumbnail_size),
                Image.Resampling.LANCZOS,
            )
            images.append(im)

        # Reprompt as necessary until a valid response is received.
        logging.info(f"Sending {len(images)} image(s) to Gemini")
        query = Query(prompt, images, {"temperature": temperature})
        response = query_with_reprompts(
            self._gemini_model,
            query,
            [FunctionalRepromptCheck(self.reprompt_and_parse)],
        )
        logging.info("Received response from Gemini")

        # Parse response
        err = self.reprompt_and_parse(query, response)
        parsed_response = response.metadata.get("parsed", None)
        if err is not None:
            raise ValueError(f"VLM response failed!\nMore details:\n{err.prompt}")
        assert (
            parsed_response is not None
        ), 'Parsed output must be stored in response.metadata["parsed"]'
        return parsed_response
