"""Base class for prompt search methods."""

import abc

from pldo.structs import PromptSearchDataset


class PromptSearch(abc.ABC):
    """Base class for prompt search methods."""

    @abc.abstractmethod
    def train(self, data: PromptSearchDataset) -> str:
        """The main learning method: returns an optimized prompt for the object."""
