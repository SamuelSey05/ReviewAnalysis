from datetime import date
import json
from types import SimpleNamespace

import numpy as np
import torch

from src.data_models import EncodedReleaseNote, EncodedReview, PairResult

from src import release_notes_comparison_plots, release_notes_comparison_utils

from src.model_architecture import AspectSentimentExtractor
from src.processing import extract_keyword_tokens


def make_encoded_release_note(
    release_note_id: str = "n1",
    aspect: int = 0,
    date_str: str = "10 January 2024",
    content: str = "fixed crash on startup",
    embedding_values: tuple[float, ...] = (1.0, 0.0),
    version: str = "1.0.0",
) -> EncodedReleaseNote:
    return EncodedReleaseNote(
        release_note_id=release_note_id,
        version=version,
        date=date_str,
        content=content,
        aspect=aspect,
        embedding=np.array(embedding_values),
        tokens=extract_keyword_tokens(content),
    )


def make_encoded_review(
    review_id: str = "r1",
    aspect: int = 0,
    at: str = "2024-01-01 00:00:00",
    content: str = "app crashes on startup",
    sentiment: int = 0,
    embedding_values: tuple[float, ...] = (1.0, 0.0),
    score: int = 1,
) -> EncodedReview:
    return EncodedReview(
        review_id=review_id,
        content=content,
        at=at,
        score=score,
        aspect=aspect,
        sentiment=sentiment,
        embedding=np.array(embedding_values),
        tokens=extract_keyword_tokens(content),
    )


def sample_release_note_sorted_results() -> list[tuple[tuple[str, str], PairResult]]:
    note1 = make_encoded_release_note("n1", aspect=0, date_str="10 January 2024", content="fix crash startup")
    note2 = make_encoded_release_note("n2", aspect=1, date_str="12 January 2024", content="improve dark theme")
    review1 = make_encoded_review("r1", aspect=0, at="2024-01-01 00:00:00", content="app crash startup", sentiment=0)
    review2 = make_encoded_review("r2", aspect=1, at="2024-01-02 00:00:00", content="dark mode needs work", sentiment=0)

    return [
        (("n1", "r1"), PairResult(
            similarity=0.9,
            release_note=note1,
            review=review1,
            longest_match_length=2,
            time_diff_days=9,
        )),
        (("n2", "r2"), PairResult(
            similarity=0.4,
            release_note=note2,
            review=review2,
            longest_match_length=1,
            time_diff_days=10,
        )),
    ]
class FakeAspectSentimentModel(AspectSentimentExtractor):
    def __init__(self):
        pass

    def encoder(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=torch.ones((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float)
        )

    def aspect_sentiment_inference(self, texts, batch_size=64):
        n = len(texts)
        return [i % 3 for i in range(n)], [2 for _ in range(n)]
    
    def get_embeddings(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        return torch.full((len(texts), 4), 0.5, dtype=torch.float)
    

def test_extract_keyword_tokens_normalizes_and_tokenizes():
    tokens = extract_keyword_tokens("App's CRASH fixed in v2.0!")
    assert tokens == ["app's", "crash", "fixed", "in", "v2", "0"]


def test_collect_encoded_rows_returns_indexed_model_outputs(monkeypatch):
    rows = [
        {"content": "first"},
        {"content": "second"},
    ]
    collected = release_notes_comparison_utils._collect_encoded_rows(FakeAspectSentimentModel(), rows)

    assert len(collected) == 2
    assert collected[0][0] == 0
    assert collected[1][0] == 1
    assert collected[0][3] == 0
    assert collected[1][4] == 2


def test_encode_release_notes_builds_typed_objects(monkeypatch):
    rows = [{
        "release_note_id": "n10",
        "version": "2.1.0",
        "date": "1 February 2024",
        "content": "fixed login issue",
    }]

    monkeypatch.setattr(
        release_notes_comparison_utils,
        "load_csv_rows",
        lambda app_name: rows,
    )

    monkeypatch.setattr(
        release_notes_comparison_utils,
        "_collect_encoded_rows",
        lambda model, rows: [
            (0, rows[0], [0.1, 0.2], 3, 1),
        ],
    )

    encoded = release_notes_comparison_utils.encode_release_notes(model=FakeAspectSentimentModel(), release_notes_path="dummy_path")

    assert 0 in encoded
    assert encoded[0].release_note_id == "n10"
    assert encoded[0].aspect == 3
    assert encoded[0].tokens == ["fixed", "login", "issue"]


def test_encode_reviews_builds_typed_objects(monkeypatch):
    rows = [{
        "reviewId": "r9",
        "content": "login keeps failing",
        "at": "2024-01-03 10:00:00",
        "score": "1",
        "replyContent": "",
        "repliedAt": "",
    }]

    monkeypatch.setattr(
        release_notes_comparison_utils,
        "load_csv_rows",
        lambda app_name: rows,
    )

    monkeypatch.setattr(
        release_notes_comparison_utils,
        "_collect_encoded_rows",
        lambda model, rows: [
            (0, rows[0], [0.3, 0.4], 2, 0),
        ],
    )

    encoded = release_notes_comparison_utils.encode_reviews(model=FakeAspectSentimentModel(), reviews_path="dummy_path")

    assert 0 in encoded
    assert encoded[0].review_id == "r9"
    assert encoded[0].sentiment == 0
    assert encoded[0].score == 1
    assert encoded[0].tokens == ["login", "keeps", "failing"]


def test_pair_dates_parses_expected_formats():
    review = make_encoded_review(at="2024-03-01 11:22:33")
    note = make_encoded_release_note(date_str="15 March 2024")

    review_date, note_date = release_notes_comparison_utils.pair_dates(review, note)
    assert review_date == date(2024, 3, 1)
    assert note_date == date(2024, 3, 15)


def test_filter_pairs_applies_threshold_and_time_window(monkeypatch):
    r_good = make_encoded_review("r_good", aspect=0, at="2024-01-01 00:00:00", sentiment=0)

    n_good = make_encoded_release_note("n_good", aspect=0, date_str="20 January 2024")
    n_low_cos = make_encoded_release_note("n_low_cos", aspect=0, date_str="20 January 2024")
    n_old = make_encoded_release_note("n_old", aspect=0, date_str="2 February 2025")

    scores = {
        ("r_good", "n_good"): 0.5,
        ("r_good", "n_low_cos"): 0.1,
        ("r_good", "n_old"): 0.5,
    }

    def fake_cos(review, note):
        return scores[(review.review_id, note.release_note_id)]

    monkeypatch.setattr(release_notes_comparison_utils, "pair_cosine_similarity", fake_cos)

    filtered = release_notes_comparison_utils.filter_pairs(
        {0: r_good},
        {0: n_good, 1: n_low_cos, 2: n_old},
    )

    assert len(filtered) == 1
    assert filtered[0][0].review_id == "r_good"
    assert filtered[0][1].release_note_id == "n_good"


def test_get_longest_match_returns_expected_tokens():
    a = ["fix", "crash", "startup"]
    b = ["please", "fix", "crash", "soon"]
    assert release_notes_comparison_utils.get_longest_match(a, b) == ["fix", "crash"]


def test_score_pairs_contains_expected_fields(monkeypatch):
    note = make_encoded_release_note("n1", content="fix crash startup", aspect=0)
    review = make_encoded_review("r1", content="fix crash now", aspect=0)

    monkeypatch.setattr(release_notes_comparison_utils, "pair_cosine_similarity", lambda n, r: 0.5)

    scored = release_notes_comparison_utils.score_pairs(FakeAspectSentimentModel(), [(review, note)])
    key = ("r1", "n1")

    assert key in scored
    assert isinstance(scored[key], PairResult)
    assert scored[key].longest_match_length >= 1
    assert scored[key].similarity >= 0


def test_format_result_rows_formats_payload_correctly():
    sorted_results = sample_release_note_sorted_results()
    rows = release_notes_comparison_utils.format_result_rows(sorted_results[:1], ["perf", "ui"], start_rank=1)

    assert rows[0]["rank"] == 1
    assert rows[0]["review_id"] == "r1"
    assert rows[0]["release_note_id"] == "n1"
    assert "(perf)" in rows[0]["release_note"]


def test_calculate_reactivity_uses_threshold_and_returns_density_mttr():
    sorted_results = sample_release_note_sorted_results()
    density, fulfilments = release_notes_comparison_utils.calculate_reactivity(sorted_results, no_of_negative_reviews=4, threshold=0.5)

    assert density == 0.25
    assert fulfilments == [9]


def test_dedup_results_by_release_note_keeps_first_occurrence():
    sorted_results = sample_release_note_sorted_results()
    sorted_results.insert(1, (("n1", "rX"), PairResult(
        similarity=0.8,
        review=make_encoded_review("rX"),
        release_note=make_encoded_release_note("n1"),
        longest_match_length=1,
        time_diff_days=11,
    )))

    deduped = release_notes_comparison_utils.dedup_results_by_release_note(sorted_results)
    ids = [pair[0][0] for pair in deduped]

    assert ids.count("n1") == 1
    assert len(deduped) == 2


def test_write_results_to_json_writes_stats_and_matches(tmp_path, monkeypatch):
    sorted_results = sample_release_note_sorted_results()
    out_path = tmp_path / "results.json"

    monkeypatch.setattr(release_notes_comparison_utils, "load_aspect_labels", lambda: ["perf", "ui", "other"])

    release_notes_comparison_utils.write_top_and_bottom_pairs_to_json(
        sorted_results=sorted_results,
        total_candidate_pairs=10,
        no_of_negative_reviews=4,
        output_file=str(out_path),
        k=1,
    )

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "top_matches" in data
    assert "bottom_matches" in data
    assert data["stats"]["total_pairs"] == 2
    assert data["stats"]["pairs_before_filter"] == 10
    assert data["stats"]["times_to_resolutions"] == [9]


def test_write_aspect_based_metrics_writes_file_and_calls_plot(tmp_path, monkeypatch):
    sorted_results = sample_release_note_sorted_results()
    out_path = tmp_path / "aspect_metrics.json"

    called = {"value": False}

    def fake_plot(aspect_metrics, aspect_labels, output_file):
        called["value"] = True

    monkeypatch.setattr(release_notes_comparison_utils, "plot_aspect_density_comparison", fake_plot)

    release_notes_comparison_utils.write_aspect_based_metrics(
        sorted_results=sorted_results,
        output_file=str(out_path),
    )

    with open(out_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "0" in metrics or 0 in metrics
    assert called["value"] is True


def test_plot_functions_create_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(release_notes_comparison_utils, "load_aspect_labels", lambda: ["perf", "ui", "other"])

    app_summaries = {
        "slack": {
            "times_to_resolutions": [10, 20, 30],
            "mean_time_to_resolution_days": 20,
        },
        "discord": {
            "times_to_resolutions": [5, 15, 25],
            "mean_time_to_resolution_days": 15,
        },
    }

    release_notes_comparison_plots.plot_mttr_comparison(app_summaries)
    release_notes_comparison_plots.plot_resolution_time_distribution(app_summaries)
    release_notes_comparison_plots.plot_aspect_density_comparison(
        {0: {"match_density": 0.5}, 1: {"match_density": 0.25}},
        aspect_labels=["perf", "ui", "other"],
    )

    assert (tmp_path / "results" / "mttr_comparison.png").exists()
    assert (tmp_path / "results" / "resolution_time_distribution.png").exists()
    assert (tmp_path / "results" / "aspect_density_comparison.png").exists()
