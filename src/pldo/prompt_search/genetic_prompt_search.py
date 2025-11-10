"""Genetic algorithm for prompt search."""

import numpy as np

from pldo.prompt_generation.base_prompt_generator import PromptGenerator
from pldo.prompt_mutation.base_prompt_mutator import PromptMutator
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
        prompt_mutator: PromptMutator,
        num_winners: int = 4,
        num_mutations: int = 15,
        num_vision_prompts: int = 1,
        num_iterations: int = 5,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self._prompt_generator = prompt_generator
        self._prompt_scorer = prompt_scorer
        self._prompt_mutator = prompt_mutator

        # Hyperparameters
        self._num_winners = num_winners
        self._num_mutations = num_mutations
        self._num_vision_prompts = num_vision_prompts
        self._num_initial_prompts = num_winners + num_mutations + num_vision_prompts
        self._num_iterations = num_iterations
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._verbose = verbose

    def train(self, data: PromptSearchDataset, stop_acc_threshold: float = 0.95) -> str:
        """Run the full genetic search process."""

        # Initial population
        initial_prompts = self._prompt_generator.generate(
            data.train_rgbs, num_samples=self._num_initial_prompts
        )

        if self._verbose:
            print("Initial prompts:")
            for i, p in enumerate(initial_prompts):
                print(f"[{i+1}] {p}")

        # Score the initial population
        scored_prompts = score_prompts(
            initial_prompts,
            data.val_positive_rgbs,
            data.val_negative_rgbs,
            self._prompt_scorer,
        )

        # Multiple iterations
        for iteration in range(self._num_iterations):
            # Rank and select winners
            ranked_prompts = sorted(
                scored_prompts, key=lambda p: p.accuracy, reverse=True
            )
            winners = ranked_prompts[: self._num_winners]

            if self._verbose:
                print(f"\n--- Iteration {iteration+1} ---")
                print("Winners:")
                for i, w in enumerate(winners):
                    print(f"[{i+1}] {w.get_table_str()}")

            # Mutate to form next generation
            mutated_prompts: list[str] = []
            for _ in range(self._num_mutations):
                selected_winners = self._sample_nonempty_prompt_subset(winners)
                mutation = self._prompt_mutator.mutate(selected_winners)
                mutated_prompts.append(mutation)

            if self._verbose:
                print("Mutated Prompts:")
                for i, p in enumerate(mutated_prompts):
                    print(f"[M{i+1}] {p}")

            # Add fresh "vision" prompts for diversity
            vision_prompts = self._prompt_generator.generate(
                data.train_rgbs, num_samples=self._num_vision_prompts
            )

            if self._verbose:
                print("Vision Prompts:")
                for i, p in enumerate(vision_prompts):
                    print(f"[V{i+1}] {p}")

            # Evaluate all new prompts
            new_prompts = mutated_prompts + vision_prompts
            new_scored_prompts = score_prompts(
                new_prompts,
                data.val_positive_rgbs,
                data.val_negative_rgbs,
                self._prompt_scorer,
            )

            # Update population
            scored_prompts = winners + new_scored_prompts
            assert len(scored_prompts) == self._num_initial_prompts

            best = max(scored_prompts, key=lambda p: p.accuracy)
            if self._verbose:
                print(
                    f"Best prompt this iteration: Accuracy: {best.accuracy:.4f} | "
                    f"Prompt: {best.prompt[:80]}..."
                )

            # Stop if the best prompt gets almost all right.
            if best.accuracy >= stop_acc_threshold:
                break

        # Return best prompt
        best_scored_prompt = max(scored_prompts, key=lambda p: p.accuracy)
        return best_scored_prompt.prompt

    def _sample_nonempty_prompt_subset(self, winners: list[ScoredPrompt]) -> list[str]:
        """Sample a non-empty random subset of winners."""
        while True:
            selected = [w.prompt for w in winners if self._rng.choice([True, False])]
            if selected:
                return selected
