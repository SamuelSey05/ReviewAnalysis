import argparse
import logging
import os
import torch

from model_entry import get_correct_model_weights_path
from src.config import DEVICE, DISTILBERT_BASE
from src.model_architecture import AspectSentimentExtractor, SBERTWrapper
from src.release_notes_comparison_plots import plot_mttr_comparison, plot_resolution_time_distribution
from src.release_notes_comparison_utils import get_app_names, release_notes_vs_reviews_comparison, write_aspect_based_metrics, write_top_and_bottom_pairs_to_json

logger = logging.getLogger(__name__)

def main() -> None:
    argparser = argparse.ArgumentParser(description="Compare release notes and reviews for an app using SBERT or aspect-based model.")
    argparser.add_argument(
        "--load_weights_from",
        type=str, 
        help="Path to model weights to load for inference (can only use with distilbert-base-uncased model) (optional)"
    )
    argparser.add_argument(
        "--use_sbert",
        action="store_true",
        help="Whether to use SBERT for the comparison instead of the aspect-based model. If not set, the aspect-based model will be used by default.",
    )
    argparser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Base directory for output files. Defaults to results/ for normal mode and results/sbert/ for SBERT mode.",
    )
    argparser.add_argument(
        "--deduplicate_results",
        action="store_true",
        help="Whether to deduplicate results for the top and bottom matches view."
    )

    args = argparser.parse_args()

    results_dir = args.results_dir if args.results_dir else ("results/sbert/" if args.use_sbert else "results/")
    logger.info(f"Results will be written to {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    if args.use_sbert:
        logger.info("Using SBERT model for comparison.")
        model = SBERTWrapper('all-MiniLM-L6-v2')
    else:
        logger.info("Using DistilBERT model for comparison.")
        model_weights_path = get_correct_model_weights_path(DISTILBERT_BASE, args.load_weights_from)
        model = AspectSentimentExtractor(DISTILBERT_BASE, num_aspects=12).to(DEVICE)

        logger.info(f"Loading model weights from {model_weights_path}...")
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        model.eval()

    app_summaries = {}

    for app_name in get_app_names():
        logger.info(f"Comparing release notes and reviews for {app_name}...")
        sorted_results, total_candidate_pairs, no_of_negative_reviews = release_notes_vs_reviews_comparison(
            model,
            app_name,
        )

        density, times_to_resolutions = write_top_and_bottom_pairs_to_json(
            sorted_results,
            total_candidate_pairs,
            no_of_negative_reviews,
            output_file=f"{results_dir}{app_name}_comparison_results.json",
            deduplicate=args.deduplicate_results,
        )
        
        write_aspect_based_metrics(
            sorted_results=sorted_results,
            output_file=f"{results_dir}{app_name}_aspect_metrics.json",
        )

        app_summaries[app_name] = {
            "total_pairs": total_candidate_pairs,
            "match_density": density,
            "times_to_resolutions": times_to_resolutions,
        }

    # Plot graphs
    plot_mttr_comparison(
        app_summaries,
        output_file=os.path.join(results_dir, "mttr_comparison.png"),
    )

    plot_resolution_time_distribution(
        app_summaries,
        output_file=os.path.join(results_dir, "resolution_time_distribution.png"),
    )

if __name__ == "__main__":
    main()