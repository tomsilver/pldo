"""Base class for prompt mutators."""

import abc


class PromptMutator(abc.ABC):
    """Base class for prompt mutators."""

    @abc.abstractmethod
    def mutate(self, winners: list[str]) -> str:
        """Mutate several unique candidate prompts for a given object."""
