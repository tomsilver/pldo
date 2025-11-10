"""Script to test object prompt creation with Gemini.

Example usage:
    python scripts/gemini_prompt_creation_demo.py \
        --init_prompt "clickshare" \
        --ref_folder "tests/pldo1/assets/clickshare/train" \
        --eval_folder "tests/pldo1/assets/clickshare/validation" \
        --use_cache_only \
        --cache_path "cache_backups" \
        --seed 123
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache

from pldo.prompt_generation.gemini_prompt_generator import GeminiPromptGenerator
from pldo.prompt_mutation.gemini_prompt_mutator import GeminiPromptMutator
from pldo.prompt_scoring.gemini_prompt_scorer import GeminiPromptScorer
from pldo.prompt_scoring.utils import ScoredPrompt, score_prompts


def _main(
    init_prompt: str,
    ref_folder: Path,
    eval_folder: Path,
    shuffle_rng: np.random.Generator,
    num_samples: int,
    use_cache_only: bool = False,
    cache_path: Path = Path(__file__).parent / "cache_backups",
) -> None:

    # --- Load reference images ---
    ref_images = [iio.imread(ref_folder / f"{i}.jpg") for i in range(5)]

    # --- Load evaluation images ---
    eval_pos_images = [iio.imread(eval_folder / "positive" / f"{i}.jpg") for i in [1]]
    eval_neg_images = [iio.imread(eval_folder / "negative" / f"{i}.jpg") for i in [1]]

    # --- Determine cache path ---
    cache_path.mkdir(parents=True, exist_ok=True)

    # --- Initialize caches ---
    gen_cache = SQLite3PretrainedLargeModelCache(cache_path / "creation_gen_cache.db")
    score_cache = SQLite3PretrainedLargeModelCache(
        cache_path / "creation_score_cache.db"
    )
    mut_cache = SQLite3PretrainedLargeModelCache(cache_path / "creation_mut_cache.db")

    # --- Initialize components ---
    prompt_generator = GeminiPromptGenerator(
        initial_prompt=init_prompt,
        cache=gen_cache,
        use_cache_only=use_cache_only,
    )
    prompt_scorer = GeminiPromptScorer(cache=score_cache, use_cache_only=use_cache_only)

    # Shuffle image order
    shuffled_imgs: list[np.ndarray] = list(shuffle_rng.permutation(ref_images))

    # Generate candidate prompts
    outputs = prompt_generator.generate(shuffled_imgs, num_samples=num_samples)
    print("Generated Prompts:")
    for i, prompt in enumerate(outputs):
        print(f"[{i}] {prompt}")

    scored_prompts: list[ScoredPrompt] = score_prompts(
        outputs, eval_pos_images, eval_neg_images, prompt_scorer
    )

    ranked_prompts = sorted(scored_prompts, key=lambda x: x.accuracy, reverse=True)

    print("\nRanked Prompts:")
    prompts: list[str] = []
    for i, entry in enumerate(ranked_prompts):
        print(f"{i+1} {entry.get_table_str()}")
        prompts += entry.prompt

    # Mutate the prompts
    prompt_mutator = GeminiPromptMutator(cache=mut_cache, use_cache_only=use_cache_only)
    mutation: str = prompt_mutator.mutate(prompts)

    scored_mutation: list[ScoredPrompt] = score_prompts(
        [mutation], eval_pos_images, eval_neg_images, prompt_scorer
    )

    print(f"M {scored_mutation[0].get_table_str()}")


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
        default=2,
        help="Number of candidate prompts to generate",
    )
    parser.add_argument(
        "--use_cache_only",
        action="store_true",
        help="Only use cached results; do not call the LLM",
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default=Path(__file__).parent / "cache_backups",
        help="Folder path to store/read SQLite caches",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Seed random
    rng = np.random.default_rng(seed=args.seed)

    _main(
        args.init_prompt,
        args.ref_folder,
        args.eval_folder,
        rng,
        args.num_samples,
        use_cache_only=args.use_cache_only,
        cache_path=args.cache_path,
    )
