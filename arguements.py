import argparse

def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "model",
        type=str,
        help="Name of the pre-trained model to use for inference."
    )

    argparser.add_argument(
        "results",
        type=str,
        help="Name of the results file to save the output."
    )

    argparser.add_argument(
        "--is_sentiment",
        action="store_true",
        help="Whether to perform sentiment analysis or aspect extraction."
    )

    argparser.add_argument(
        "--fine_tune_model",
        action="store_true",
        help="Whether to fine-tune the model before inference."
    )

    argparser.add_argument(
        "--optimize_hyperparameters",
        action="store_true",
        help="Whether to optimize hyperparameters during fine-tuning."
    )

    argparser.add_argument(
        "--no_training",
        action="store_true",
        help="Whether to skip the training phase and directly run inference with the specified model."
    )

    args = argparser.parse_args()

    return args