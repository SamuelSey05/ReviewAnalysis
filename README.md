# ReviewAnalysis

This repository implements Aspect Based Sentiment Analysis, and uses it to map user reviews to developer release notes, allowing for the calculation of metrics based on how software companies fulfil user needs.

## Project Structure

- `src/`: Source code containing model architectures, training and processing utils.
- `experiments/`: Scripts and datasets used for manual evaluation of model, using human annotation.
- `tests/`: Unit and integration tests.
- `datasets/`: Directory holding training and review/release note datasets.
- `resources/`: Directory holding list of aspect labels.
- `model_entry.py`: Entrypoint for interaction with the model, including training and inference.
- `release_notes_comparison_entry.py`: Main script for comparing release reviews and release notes across apps.

## Setup

**Prerequisites**:

- Python 3.14
- `pip`

To install dependencies run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Entry Points

1. Model Operations

    - Train the Model:

    To train the aspect and sentiment heads using the AWARE dataset:

    ```bash
    python model_entry.py --mode train
    ```

    - Run inference on the AWARE dataset

    ```bash
    python model_entry.py
    ```

    A report of accuracy scores will be put in the results dictionary, the destination can be overridden using the `--results` flag.

    - Run inference on a single text

    To run inference on a single text stored in a text file:

    ```bash
    python model_entry.py --single_review_file review_file.txt
    ```

    For any of these operations, the model can be overridden from DistilBERT base using the `--model` flag, or weights can be loaded from a pre-trained `AspectSentimentExtractor` model using the `--load_weights_from` flag.

2. Review and Release Notes Analysis

    To run the comparison script:

    ```bash
    python release_notes_comparison_entry.py
    ```

    The same `--load_weights_from` flag can be used as with the model entry point. `--results_dir` can be set to point results at a particular directory. `--deduplicate_results` limits to 1 match per release note in results. `--use_sbert` switches the model to use SentenceBERT, which is used for model comparison in my extension.

## Usage

Currently using Hugging Face without authentication, which works when only using public models, to set an auth token, use the `HF_TOKEN` environment variable.

## Testing

The suite uses `pytest` using the command:

```bash
pytest
```
