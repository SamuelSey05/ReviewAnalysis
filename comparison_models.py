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