"""Genetic algorithm for prompt search."""

from pldo.prompt_generation.base_prompt_generator import PromptGenerator
from pldo.prompt_scoring.base_prompt_scorer import PromptScorer
from pldo.prompt_search.base_prompt_search import PromptSearch
from pldo.structs import PromptSearchDataset


class GeneticPromptSearch(PromptSearch):
    """Genetic algorithm for prompt search."""

    def __init__(
        self,
        prompt_generator: PromptGenerator,
        prompt_scorer: PromptScorer,
        num_winners: int = 4,
        num_mutations: int = 15,
        num_vision_prompts: int = 1,
    ) -> None:
        self._prompt_generator = prompt_generator
        self._prompt_scorer = prompt_scorer
        # Hyperparameters.
        self._num_winners = num_winners
        self._num_mutations = num_mutations
        self._num_vision_prompts = num_vision_prompts
        self._num_initial_prompts = num_winners + num_mutations + num_vision_prompts

    def train(self, data: PromptSearchDataset) -> str:
        # First, generate initial candidate prompts.
        initial_prompts = self._prompt_generator.generate(
            data.train_rgbs, num_samples=self._num_initial_prompts
        )
