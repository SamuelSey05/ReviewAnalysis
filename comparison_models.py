from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EncodedReleaseNote:
    release_note_id: str
    version: str
    date: str
    content: str
    embedding: np.ndarray
    aspect: int
    tokens: list[str]


@dataclass(slots=True)
class EncodedReview:
    review_id: str
    content: str
    at: str
    score: int
    reply_content: str
    replied_at: str
    embedding: np.ndarray
    aspect: int
    tokens: list[str]
    sentiment: int


@dataclass(slots=True)
class PairResult:
    similarity: float
    release_note: EncodedReleaseNote
    review: EncodedReview
    lcs_length: int
    time_diff_days: int


PairKey = tuple[str, str]
NoteReviewPair = tuple[EncodedReleaseNote, EncodedReview]
RankedResult = tuple[PairKey, PairResult]
