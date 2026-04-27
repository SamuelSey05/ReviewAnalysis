from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingReview:
    """Container for one review row used during training and preprocessing.
    """

    domain: str
    app: str
    review_id: str
    title: str
    review: str
    rating: int
    is_opinion: bool

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "TrainingReview":
        """Extract a review from a CSV row dictionary.

        Args:
            data (dict[str, str]): Dictionary representing a row from the CSV file, with keys corresponding to column names.

        Returns:
            TrainingReview: TrainingReview object made with data from input dictionary.
        """

        rating = data.get("rating")

        return cls(
            domain=data.get("domain", ""),
            app=data.get("app", ""),
            review_id=data.get("review_id", ""),
            title=data.get("title", ""),
            review=data.get("review", ""),
            rating=int(rating) if rating else 0,
            is_opinion=data.get("is_opinion") == "TRUE",
        )


@dataclass
class TrainingOpinion:
    """Container for one sentence-level aspect annotation linked to a review.
    """

    review: TrainingReview
    sentence_id: str
    category: str
    term: str
    from_word: int | None
    to_word: int | None
    sentiment: str

    @classmethod
    def from_review_and_dict(cls, review: TrainingReview, data: dict[str, str]) -> "TrainingOpinion":
        """Extract a sentence from a CSV row dictionary and its parent review.

        Args:
            review (TrainingReview): TrainingReview object representing the parent review of this opinion annotation.
            data (dict[str, str]): Dictionary representing a row from the CSV file, with keys corresponding to column names.

        Returns:
            TrainingOpinion: TrainingOpinion object made with data from input dictionary and parent review.
        """

        from_value = data.get("from", "")
        to_value = data.get("to", "")

        return cls(
            review=review,
            sentence_id=data.get("sentence_id", ""),
            category=data.get("category", ""),
            term=data.get("term", ""),
            from_word=int(from_value) if from_value.isdigit() else None,
            to_word=int(to_value) if to_value.isdigit() else None,
            sentiment=data.get("sentiment", ""),
        )

    

@dataclass(slots=True)
class EncodedReleaseNote:
    """Container for one release note with its content, metadata, encoded representation and aspect label.
    """
    release_note_id: str
    version: str
    date: str
    content: str
    embedding: np.ndarray
    aspect: int
    tokens: list[str]


@dataclass(slots=True)
class EncodedReview:
    """Container for one review with its content, metadata, encoded representation and aspect/sentiment labels.
    """
    review_id: str
    content: str
    at: str
    score: int
    embedding: np.ndarray
    aspect: int
    tokens: list[str]
    sentiment: int


@dataclass(slots=True)
class PairResult:
    """Container for a (review, note) pair with similarity, longest match length and time difference
    """
    similarity: float
    review: EncodedReview
    release_note: EncodedReleaseNote
    longest_match_length: int
    time_diff_days: int


PairKey = tuple[str, str]
ReviewNotePair = tuple[EncodedReview, EncodedReleaseNote]
RankedResult = tuple[PairKey, PairResult]
