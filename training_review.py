from dataclasses import dataclass


@dataclass
class TrainingReview:
    domain: str
    app: str
    review_id: str
    title: str
    review: str
    rating: int
    is_opinion: bool

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "TrainingReview":
        """Extract a review from a CSV row dictionary."""

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
class Sentence:
    review: TrainingReview
    sentence_id: str
    category: str
    term: str
    from_word: int | None
    to_word: int | None
    sentiment: str

    @classmethod
    def from_review_and_dict(cls, review: TrainingReview, data: dict[str, str]) -> "Sentence":
        """Extract a sentence from a CSV row dictionary and its parent review."""

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

    