class Review:
    def __init__(self, domain: str, app:str, review_id:str, sentence_id:str, title:str, review:str, rating:int, is_opinion:bool):
        self.domain = domain
        self.app = app
        self.review_id = review_id
        self.sentence_id = sentence_id
        self.title = title
        self.review = review
        self.rating = rating
        self.is_opinion = is_opinion

    @staticmethod
    def from_dict(data: dict[str, str]) -> 'Review':
        return Review(
            domain=data.get("domain"),
            app=data.get("app"),
            review_id=data.get("review_id"),
            sentence_id=data.get("sentence_id"),
            title=data.get("title"),
            review=data.get("review"),
            rating=int(data.get("rating")),
            is_opinion=data.get("is_opinion")=="TRUE"
        )

class Sentence:
    def __init__(self, review: Review, category:str, term:str, from_word:int|None, to_word:int|None, sentiment:str):
        self.review = review
        self.category = category
        self.term = term
        self.from_word = from_word
        self.to_word = to_word
        self.sentiment = sentiment

    @staticmethod
    def from_review_and_dict(review: Review, data: dict[str, str]) -> 'Sentence':
        return Sentence(
            review=review,
            category=data.get("category"),
            term=data.get("term"),
            from_word=int(data.get("from")) if data.get("from").isdigit() else None,
            to_word=int(data.get("to")) if data.get("to").isdigit() else None,
            sentiment=data.get("sentiment")
        )

    