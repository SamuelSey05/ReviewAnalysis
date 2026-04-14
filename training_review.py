class TrainingReivew:
    def __init__(self, domain: str, app:str, review_id:str, title:str, review:str, rating:int, is_opinion:bool):
        """Class representing a review entry from the AWARE dataset

        Args:
            domain (str): Which domain the app belongs to
            app (str): App name
            review_id (str): UUID of the review
            title (str): Title of the review
            review (str): Text content of the reveiw
            rating (int): Star rating of the review
            is_opinion (bool): AWARE flag for if the review has an opinion
        """
        self.domain = domain
        self.app = app
        self.review_id = review_id
        self.title = title
        self.review = review
        self.rating = rating
        self.is_opinion = is_opinion

    @staticmethod
    def from_dict(data: dict[str, str]) -> 'TrainingReivew':
        """Extracts review from dictionary taken from csv

        Args:
            data (dict[str, str]): Data dictionary from csv row

        Returns:
            Review: Review extracted from the dictionary
        """
        return TrainingReivew(
            domain=data.get("domain"),
            app=data.get("app"),
            review_id=data.get("review_id"),
            title=data.get("title"),
            review=data.get("review"),
            rating=int(data.get("rating")),
            is_opinion=data.get("is_opinion")=="TRUE"
        )

class Sentence:
    def __init__(self, review: TrainingReivew, sentence_id:str, category:str, term:str, from_word:int, to_word:int, sentiment:str):
        """Class representing a sentence from the AWARE dataset, which is part of a review with a sentiment

        Args:
            review (Review): Review the sentence is part of
            sentence_id (str): UUID of the sentence
            category (str): Category the sentence discusses
            term (str): Specific term the sentence discusses
            from_word (int): Start index of the relevant part of the review text
            to_word (int): End index of the relevant part of the review text
            sentiment (str): Sentiment of the sentence (Positive, Negative) gathered by AWARE annotators
        """
        self.review = review
        self.sentence_id = sentence_id
        self.category = category
        self.term = term
        self.from_word = from_word
        self.to_word = to_word
        self.sentiment = sentiment

    @staticmethod
    def from_review_and_dict(review: TrainingReivew, data: dict[str, str]) -> 'Sentence':
        """Extracts sentence from dictionary taken from csv

        Args:
            review (Review): Review the sentence is part of
            data (dict[str, str]): Data dictionary from csv row

        Returns:
            Sentence: Sentence extracted from the dictionary and review
        """
        return Sentence(
            review=review,
            sentence_id=data.get("sentence_id"),
            category=data.get("category"),
            term=data.get("term"),
            from_word=int(data.get("from")) if data.get("from").isdigit() else None,
            to_word=int(data.get("to")) if data.get("to").isdigit() else None,
            sentiment=data.get("sentiment")
        )

    