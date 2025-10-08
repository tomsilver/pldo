"""Genetic algorithm for prompt search."""

import numpy as np

from pldo.prompt_generation.base_prompt_generator import PromptGenerator
from pldo.prompt_scoring.base_prompt_scorer import PromptScorer
from pldo.prompt_scoring.utils import ScoredPrompt, score_prompts
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
        num_iterations: int = 5,
        seed: int = 0,
    ) -> None:
        self._prompt_generator = prompt_generator
        self._prompt_scorer = prompt_scorer
        # Hyperparameters.
        self._num_winners = num_winners
        self._num_mutations = num_mutations
        self._num_vision_prompts = num_vision_prompts
        self._num_initial_prompts = num_winners + num_mutations + num_vision_prompts
        self._num_iterations = num_iterations
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def train(self, data: PromptSearchDataset) -> str:
        # First, generate initial candidate prompts.
        initial_prompts = self._prompt_generator.generate(
            data.train_rgbs, num_samples=self._num_initial_prompts
        )
        # Score the prompts.
        scored_prompts = score_prompts(
            initial_prompts,
            data.val_positive_rgbs,
            data.val_negative_rbgs,
            self._prompt_scorer,
        )
        # Iterate.
        for _ in range(self._num_iterations):
            # Sort and pick winners.
            ranked_prompts = sorted(
                scored_prompts, key=lambda p: p.accuracy, reverse=True
            )
            winners = ranked_prompts[: self._num_winners]
            mutated_prompts: list[str] = []
            # Get mutations.
            for _ in range(self._num_mutations):
                # Sample a random subset of the winners to condition the mutation on.
                while True:  # make sure we sample a non-empty set
                    selected_winners = []
                    for winner in winners:
                        if self._rng.choice(2):
                            selected_winners.append(winner)
                    if selected_winners:
                        break
                # Sample a mutation conditioned on the selected winners.
                mutation = self._mutate_prompts(selected_winners)
                mutated_prompts.append(mutation)
            # Create the remaining vision prompts.
            vision_prompts = self._prompt_generator.generate(
                data.train_rgbs, num_samples=self._num_vision_prompts
            )
            # Score all the new prompts.
            new_prompts = mutated_prompts + vision_prompts
            new_scored_prompts = score_prompts(
                new_prompts,
                data.val_positive_rgbs,
                data.val_negative_rbgs,
                self._prompt_scorer,
            )
            # Update scored prompts.
            scored_prompts = winners + new_scored_prompts
            assert len(scored_prompts) == self._num_initial_prompts

        # Return the best seen prompt.
        best_scored_prompt = max(scored_prompts, key=lambda p: p.accuracy)
        return best_scored_prompt.prompt

    def _mutate_prompts(self, selected_winners: list[ScoredPrompt]) -> str:
        """Mutate a new prompt given a subset of winners."""
        # TODO
        import ipdb

        ipdb.set_trace()
