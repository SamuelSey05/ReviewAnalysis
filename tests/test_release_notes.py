from datetime import date
import json
from types import SimpleNamespace

import numpy as np
import torch

from aspect_based import AspectSentimentExtractor
from comparison_models import EncodedReleaseNote, EncodedReview, PairResult
import release_notes
import release_notes_plots


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
        tokens=release_notes.extract_keyword_tokens(content),
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
        reply_content="",
        replied_at="",
        aspect=aspect,
        sentiment=sentiment,
        embedding=np.array(embedding_values),
        tokens=release_notes.extract_keyword_tokens(content),
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
            lcs_length=2,
            time_diff_days=9,
        )),
        (("n2", "r2"), PairResult(
            similarity=0.4,
            release_note=note2,
            review=review2,
            lcs_length=1,
            time_diff_days=10,
        )),
    ]
class DummyAspectSentimentModel(AspectSentimentExtractor):
    def __init__(self):
        pass

    def encoder(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=torch.ones((input_ids.shape[0], input_ids.shape[1], 4), dtype=torch.float)
        )

    def aspect_sentiment_inference(self, input_ids, attention_masks, batch_size=64):
        n = input_ids.shape[0]
        return [i % 3 for i in range(n)], [2 for _ in range(n)]


def fake_tokenize(texts, model_name):
    n = len(texts)
    return {
        "input_ids": torch.ones((n, 3), dtype=torch.long),
        "attention_mask": torch.ones((n, 3), dtype=torch.long),
    }


def fake_pool_embeddings(embeddings, attention_mask):
    return torch.full((embeddings.shape[0], 4), 0.5, dtype=torch.float)


def test_extract_keyword_tokens_normalizes_and_tokenizes():
    tokens = release_notes.extract_keyword_tokens("App's CRASH fixed in v2.0!")
    assert tokens == ["app's", "crash", "fixed", "in", "v2", "0"]


def test_collect_encoded_rows_returns_indexed_model_outputs(monkeypatch):
    monkeypatch.setattr(release_notes, "tokenize", fake_tokenize)
    monkeypatch.setattr(release_notes, "pool_embeddings", fake_pool_embeddings)

    rows = [
        {"content": "first"},
        {"content": "second"},
    ]
    collected = release_notes._collect_encoded_rows(DummyAspectSentimentModel(), rows)

    assert len(collected) == 2
    assert collected[0][0] == 0
    assert collected[1][0] == 1
    assert collected[0][3] == 0
    assert collected[1][4] == 2


def test_encode_release_notes_builds_typed_objects(monkeypatch):
    monkeypatch.setattr(
        release_notes,
        "_collect_encoded_rows",
        lambda model, rows: [
            (0, rows[0], [0.1, 0.2], 3, 1),
        ],
    )

    rows = [{
        "release_note_id": "n10",
        "version": "2.1.0",
        "date": "1 February 2024",
        "content": "fixed login issue",
    }]

    encoded = release_notes.encode_release_notes(model=DummyAspectSentimentModel(), rows=rows)

    assert 0 in encoded
    assert encoded[0].release_note_id == "n10"
    assert encoded[0].aspect == 3
    assert encoded[0].tokens == ["fixed", "login", "issue"]


def test_encode_reviews_builds_typed_objects(monkeypatch):
    monkeypatch.setattr(
        release_notes,
        "_collect_encoded_rows",
        lambda model, rows: [
            (0, rows[0], [0.3, 0.4], 2, 0),
        ],
    )

    rows = [{
        "reviewId": "r9",
        "content": "login keeps failing",
        "at": "2024-01-03 10:00:00",
        "score": "1",
        "replyContent": "",
        "repliedAt": "",
    }]

    encoded = release_notes.encode_reviews(model=DummyAspectSentimentModel(), rows=rows)

    assert 0 in encoded
    assert encoded[0].review_id == "r9"
    assert encoded[0].sentiment == 0
    assert encoded[0].score == 1
    assert encoded[0].tokens == ["login", "keeps", "failing"]


def test_pair_dates_parses_expected_formats():
    note = make_encoded_release_note(date_str="15 March 2024")
    review = make_encoded_review(at="2024-03-01 11:22:33")

    note_date, review_date = release_notes.pair_dates(note, review)
    assert note_date == date(2024, 3, 15)
    assert review_date == date(2024, 3, 1)


def test_filter_pairs_applies_threshold_and_time_window(monkeypatch):
    n_good = make_encoded_release_note("n_good", aspect=0, date_str="20 January 2024")
    n_low_cos = make_encoded_release_note("n_low_cos", aspect=0, date_str="20 January 2024")
    n_old = make_encoded_release_note("n_old", aspect=0, date_str="2 February 2025")

    r_good = make_encoded_review("r_good", aspect=0, at="2024-01-01 00:00:00", sentiment=0)

    scores = {
        ("n_good", "r_good"): 0.5,
        ("n_low_cos", "r_good"): 0.1,
        ("n_old", "r_good"): 0.5,
    }

    def fake_cos(note, review):
        return scores[(note.release_note_id, review.review_id)]

    monkeypatch.setattr(release_notes, "pair_cosine_similarity", fake_cos)

    filtered = release_notes.filter_pairs(
        {0: n_good, 1: n_low_cos, 2: n_old},
        {0: r_good},
    )

    assert len(filtered) == 1
    assert filtered[0][0].release_note_id == "n_good"
    assert filtered[0][1].review_id == "r_good"


def test_get_longest_match_returns_expected_tokens():
    a = ["fix", "crash", "startup"]
    b = ["please", "fix", "crash", "soon"]
    assert release_notes.get_longest_match(a, b) == ["fix", "crash"]


def test_score_pairs_contains_expected_fields(monkeypatch):
    note = make_encoded_release_note("n1", content="fix crash startup", aspect=0)
    review = make_encoded_review("r1", content="fix crash now", aspect=0)

    monkeypatch.setattr(release_notes, "pair_cosine_similarity", lambda n, r: 0.5)

    scored = release_notes.score_pairs([(note, review)])
    key = ("n1", "r1")

    assert key in scored
    assert isinstance(scored[key], PairResult)
    assert scored[key].lcs_length >= 1
    assert scored[key].similarity >= 0


def test_format_result_rows_formats_payload_correctly():
    sorted_results = sample_release_note_sorted_results()
    rows = release_notes.format_result_rows(sorted_results[:1], ["perf", "ui"], start_rank=1)

    assert rows[0]["rank"] == 1
    assert rows[0]["release_note_id"] == "n1"
    assert rows[0]["review_id"] == "r1"
    assert "(perf)" in rows[0]["release_note"]


def test_calculate_reactivity_uses_threshold_and_returns_density_mttr():
    sorted_results = sample_release_note_sorted_results()
    density, mttr = release_notes.calculate_reactivity(sorted_results, no_of_negative_reviews=4, threshold=0.5)

    assert density == 0.25
    assert mttr == 9


def test_dedup_results_by_release_note_keeps_first_occurrence():
    sorted_results = sample_release_note_sorted_results()
    sorted_results.insert(1, (("n1", "rX"), PairResult(
        similarity=0.8,
        release_note=make_encoded_release_note("n1"),
        review=make_encoded_review("rX"),
        lcs_length=1,
        time_diff_days=11,
    )))

    deduped = release_notes.dedup_results_by_release_note(sorted_results)
    ids = [pair[0][0] for pair in deduped]

    assert ids.count("n1") == 1
    assert len(deduped) == 2


def test_write_results_to_json_writes_stats_and_matches(tmp_path, monkeypatch):
    sorted_results = sample_release_note_sorted_results()
    out_path = tmp_path / "results.json"

    monkeypatch.setattr(release_notes, "load_aspect_labels", lambda: ["perf", "ui", "other"])

    release_notes.write_results_to_json(
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

    monkeypatch.setattr(release_notes_plots, "plot_aspect_density_comparison", fake_plot)

    release_notes.write_aspect_based_metrics(
        sorted_results=sorted_results,
        aspect_counts={0: 2, 1: 1},
        output_file=str(out_path),
    )

    with open(out_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "0" in metrics or 0 in metrics
    assert called["value"] is True


def test_write_results_creates_all_output_variants(tmp_path, monkeypatch):
    sorted_results = sample_release_note_sorted_results()
    base_path = tmp_path / "workflow.json"

    monkeypatch.setattr(release_notes, "load_aspect_labels", lambda: ["perf", "ui", "other"])

    release_notes.write_results(
        sorted_results=sorted_results,
        total_candidate_pairs=12,
        no_of_negative_reviews=4,
        aspect_counts={0: 2, 1: 1},
        output_file=str(base_path),
        k=1,
    )

    assert base_path.exists()
    assert (tmp_path / "workflow_dedup.json").exists()
    assert (tmp_path / "workflow_aspect_metrics.json").exists()


def test_plot_functions_create_output_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(release_notes, "load_aspect_labels", lambda: ["perf", "ui", "other"])

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

    all_results = sample_release_note_sorted_results()

    release_notes_plots.plot_mttr_comparison(app_summaries)
    release_notes_plots.plot_resolution_time_distribution(app_summaries)
    release_notes_plots.plot_specificity_comparison(all_results)
    release_notes_plots.plot_aspect_density_comparison(
        {0: {"match_density": 0.5}, 1: {"match_density": 0.25}},
        aspect_labels=["perf", "ui", "other"],
    )

    assert (tmp_path / "results" / "mttr_comparison.png").exists()
    assert (tmp_path / "results" / "resolution_time_distribution.png").exists()
    assert (tmp_path / "results" / "similarity_vs_resolution_time.png").exists()
    assert (tmp_path / "results" / "aspect_density_comparison.png").exists()
