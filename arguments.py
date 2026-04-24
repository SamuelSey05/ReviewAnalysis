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
        "--no_training",
        action="store_true",
        help="Whether to skip the training phase and directly run inference with the specified model."
    )

    args = argparser.parse_args()

    return args