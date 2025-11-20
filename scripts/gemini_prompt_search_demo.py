"""Script to test genetic prompt search with Gemini.

Example usage:
    python scripts/gemini_prompt_search_demo.py \
        --init_prompt "clickshare" \
        --ref_folder "tests/pldo1/assets/clickshare/train" \
        --eval_folder "tests/pldo1/assets/clickshare/validation" \
        --use_cache_only \
        --eval_count 10 \
        --cache_path "scripts/cache_backups"
"""

from pathlib import Path

import imageio.v3 as iio
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache

from pldo.prompt_generation.gemini_prompt_generator import GeminiPromptGenerator
from pldo.prompt_mutation.gemini_prompt_mutator import GeminiPromptMutator
from pldo.prompt_scoring.gemini_prompt_scorer import GeminiPromptScorer
from pldo.prompt_search.genetic_prompt_search import GeneticPromptSearch
from pldo.structs import PromptSearchDataset


def _main(
    init_prompt: str,
    ref_folder: Path,
    eval_folder: Path,
    cache_path: Path,
    use_cache_only: bool = False,
    eval_count: int = 3,
    num_winners: int = 3,
    num_mutations: int = 4,
    num_vision_prompts: int = 1,
    num_iterations: int = 4,
    seed: int = 42,
) -> None:

    # --- Load reference images ---
    ref_images = [iio.imread(ref_folder / f"{i}.jpg") for i in range(5)]

    # --- Load evaluation images ---
    eval_pos_images = [
        iio.imread(eval_folder / "positive" / f"{i}.jpg") for i in range(eval_count)
    ]
    eval_neg_images = [
        iio.imread(eval_folder / "negative" / f"{i}.jpg") for i in range(eval_count)
    ]

    # --- Ensure cache directory exists ---
    cache_path.mkdir(parents=True, exist_ok=True)

    # --- Initialize caches ---
    gen_cache = SQLite3PretrainedLargeModelCache(cache_path / "search_gen_cache.db")
    score_cache = SQLite3PretrainedLargeModelCache(cache_path / "search_score_cache.db")
    mut_cache = SQLite3PretrainedLargeModelCache(cache_path / "search_mut_cache.db")

    # --- Initialize components ---
    prompt_generator = GeminiPromptGenerator(
        initial_prompt=init_prompt,
        cache=gen_cache,
        use_cache_only=use_cache_only,
    )
    prompt_scorer = GeminiPromptScorer(
        cache=score_cache,
        use_cache_only=use_cache_only,
    )
    prompt_mutator = GeminiPromptMutator(
        cache=mut_cache,
        use_cache_only=use_cache_only,
    )

    # --- Wrap in dataset object ---
    dataset = PromptSearchDataset(
        train_rgbs=ref_images,
        val_positive_rgbs=eval_pos_images,
        val_negative_rgbs=eval_neg_images,
    )

    # --- Initialize GeneticPromptSearch ---
    genetic_search = GeneticPromptSearch(
        prompt_generator=prompt_generator,
        prompt_scorer=prompt_scorer,
        prompt_mutator=prompt_mutator,
        num_winners=num_winners,
        num_mutations=num_mutations,
        num_vision_prompts=num_vision_prompts,
        num_iterations=num_iterations,
        seed=seed,
        verbose=True,
    )

    # --- Run the search ---
    best_prompt = genetic_search.train(dataset)
    print("\n=== Best Prompt Found ===")
    print(best_prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo for Genetic Prompt Search with Gemini"
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
        "--cache_path",
        type=Path,
        default=Path(__file__).parent / "cache_backups",
        help="Folder path to store/read SQLite caches",
    )
    parser.add_argument(
        "--use_cache_only",
        action="store_true",
        help="Only use cached results; do not call the LLM",
    )

    # --- Numeric hyperparameters ---
    parser.add_argument(
        "--eval_count",
        type=int,
        default=10,
        help="Number of evaluation images to use from each (positive/negative) set",
    )
    parser.add_argument(
        "--num_winners",
        type=int,
        default=3,
        help="Number of winning prompts to select in each generation",
    )
    parser.add_argument(
        "--num_mutations",
        type=int,
        default=4,
        help="Number of mutations to apply to prompts",
    )
    parser.add_argument(
        "--num_vision_prompts",
        type=int,
        default=1,
        help="Number of prompts to generate per vision input",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=4,
        help="Number of iterations to run the genetic search",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    _main(
        args.init_prompt,
        args.ref_folder,
        args.eval_folder,
        cache_path=args.cache_path,
        use_cache_only=args.use_cache_only,
        eval_count=args.eval_count,
        num_winners=args.num_winners,
        num_mutations=args.num_mutations,
        num_vision_prompts=args.num_vision_prompts,
        num_iterations=args.num_iterations,
        seed=args.seed,
    )
