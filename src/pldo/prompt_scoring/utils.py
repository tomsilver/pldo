"""Shared utility functions."""

from dataclasses import dataclass

from prpl_perception_utils.structs import RGBImage

from pldo.prompt_scoring.base_prompt_scorer import PromptScorer

# Constants for terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


@dataclass(frozen=True)
class ScoredPrompt:
    """An individual scored prompt."""

    prompt: str
    tp: int  # true positives
    fp: int  # false positives
    fn: int  # false negatives
    tn: int  # true negatives

    def get_table_str(self) -> str:
        """Get a nice table of the confusion matrix and accuracy and prompt."""
        s = f"Accuracy: {self.accuracy} | Prompt: {self.prompt}\n"
        s += (
            f"    TP: {GREEN}{self.tp}{RESET} | "
            f"FP: {RED}{self.fp}{RESET} | "
            f"FN: {RED}{self.fn}{RESET} | "
            f"TN: {GREEN}{self.tn}{RESET}"
        )

        return s

    @property
    def total_num(self) -> float:
        """The total number of examples."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        """The accuracy."""
        return (self.tp + self.tn) / self.total_num


def score_prompts(
    prompts: list[str],
    pos_images: list[RGBImage],
    neg_images: list[RGBImage],
    prompt_scorer: PromptScorer,
) -> list[ScoredPrompt]:
    """Score and rank candidate prompts on positive and negative images."""
    scored_prompts: list[ScoredPrompt] = []

    for prompt in prompts:
        pos_scores = []
        neg_scores = []

        TP = FP = FN = TN = 0  # Initialize counters

        # Evaluate on positive images (ground truth = positive)
        for img in pos_images:
            score = prompt_scorer.score(img, prompt, negative=False)
            pos_scores.append(score)

            if score == 1:
                TP += 1
            else:
                FN += 1

        # Evaluate on negative images (ground truth = negative)
        for img in neg_images:
            score = prompt_scorer.score(img, prompt, negative=True)
            neg_scores.append(score)

            if score == 1:
                TN += 1
            else:
                FP += 1

        scored_prompts.append(ScoredPrompt(prompt=prompt, tp=TP, fp=FP, fn=FN, tn=TN))

    return scored_prompts
