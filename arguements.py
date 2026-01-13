import argparse

def get_args():
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

    args = argparser.parse_args()

    return args