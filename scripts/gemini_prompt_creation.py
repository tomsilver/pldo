"""Script to test object prompt creation with Gemini.

Example usage:
    python scripts/gemini_prompt_creation.py \
        --init_prompt "clickshare" \
        --ref_folder "tests/pldo1/assets/clickshare/train" \
        --eval_folder "tests/pldo1/assets/clickshare/validation"
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pldo.prompt_generation.gemini_prompt_generator import GeminiPromptGenerator
from pldo.prompt_scoring.gemini_prompt_scorer import GeminiPromptScorer
from pldo.prompt_scoring.utils import score_prompts


def _main(
    init_prompt: str,
    ref_folder: Path,
    eval_folder: Path,
    shuffle_rng: np.random.Generator,
    num_samples: int,
) -> None:

    # Load reference images once
    ref_images = [iio.imread(ref_folder / f"{i}.jpg") for i in range(5)]

    # Load eval images once
    eval_pos_images = [iio.imread(eval_folder / "positive" / f"{i}.jpg") for i in [1]]
    eval_neg_images = [iio.imread(eval_folder / "negative" / f"{i}.jpg") for i in [1]]

    # Initialize generator
    prompt_generator = GeminiPromptGenerator(initial_prompt=init_prompt)

    # Initialize scorer
    prompt_scorer = GeminiPromptScorer()

    # Shuffle image order
    shuffled_imgs: list[np.ndarray] = list(shuffle_rng.permutation(ref_images))

    # Generate candidate prompts from all reference images in one go
    outputs = prompt_generator.generate(shuffled_imgs, num_samples=num_samples)
    print("Generated Prompts:")
    for i, prompt in enumerate(outputs):
        print(f"[{i}] {prompt}")

    scored_prompts = score_prompts(
        outputs, eval_pos_images, eval_neg_images, prompt_scorer
    )

    # Sort prompts by final_score descending
    ranked_prompts = sorted(scored_prompts, key=lambda x: x.accuracy, reverse=True)

    print("\nRanked Prompts:")
    for i, entry in enumerate(ranked_prompts):
        print(f"{i+1} {entry.get_table_str()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test object prompt creation with Gemini"
    )
    parser.add_argument(
        "--init_prompt", type=str, required=True, help="An initial prompt"
    )
    parser.add_argument(
        "--ref_folder", type=Path, required=True, help="Reference folder path"
    )
    parser.add_argument(
        "--eval_folder", type=Path, required=True, help="Evaluation folder path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=2,
        help="Evaluation folder path",
    )

    args = parser.parse_args()

    # Seed random
    rng = np.random.default_rng(seed=42)

    _main(args.init_prompt, args.ref_folder, args.eval_folder, rng, args.num_samples)
