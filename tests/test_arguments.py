import sys

from arguements import get_args

def test_get_args_parses_no_training_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "test_model_name", "test_out_file", "--no_training"]
    )

    args = get_args()

    assert args.model == "test_model_name"
    assert args.results == "test_out_file"
    assert args.no_training is True
    assert args.is_sentiment is False
    assert args.fine_tune_model is False
    assert args.optimize_hyperparameters is False