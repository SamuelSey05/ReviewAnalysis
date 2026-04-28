from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import DEVICE
from src.processing import pool_embeddings

class AspectSentimentModel(ABC):
    @abstractmethod
    def aspect_sentiment_inference(self, texts: list[str], batch_size: int = 64) -> tuple[list[int], list[int]]:
        """Inference method for aspect and sentiment extraction. Gives predictions for aspect and sentiment classes.
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Get embeddings from the encoder for the given text.
        """
        pass

class AspectSentimentExtractor(AspectSentimentModel, torch.nn.Module):
    def __init__(self, model_name:str,  num_aspects: int, num_sentiments: int = 3) -> None: 
        """Initialize aspect/sentiment extractor with DistilBERT encoder.

        Encoder is frozen by default, with only the last 2 transformer layers unfrozen for fine-tuning.

        Args:
            model_name (str): Name of the model to load from Hugging Face (e.g., 'distilbert-base-uncased').
            num_aspects (int): Number of aspects to classify.
            num_sentiments (int, optional): Number of sentiment classes. Defaults to 3.
        """

        super().__init__()

        try:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        except OSError as e:
            raise RuntimeError(f"Could not load model '{model_name}': {e}")
        
        ## To unfreeeze all encoder parameters, uncomment the following lines:
        # for param in self.encoder.parameters():
        #     param.requires_grad = True

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 layers of encoder parameters to be fine tuned
        for layer in self.encoder.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(0.1)
        self.aspect_head = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_aspects)
        )
        self.sentiment_head = torch.nn.Linear(768, num_sentiments)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        """Forward pass for aspect and sentiment extraction.

        Args:
            input_ids (torch.Tensor): Token IDs tensor of shape (batch_size, seq_length).
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing aspect logits and sentiment logits tensors.
        """

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        mean_pooled = pool_embeddings(embeddings, attention_mask)

        x = self.dropout(mean_pooled)  # Use mean pooled representation
        aspect_logits = self.aspect_head(x)
        sentiment_logits = self.sentiment_head(x)
        return aspect_logits, sentiment_logits

    def tokenise(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise inputted texts

        Args:
            texts (list[str]): Texts to be tokenised

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input IDs and attention masks for the tokenised texts
        """

        tokenised = self.tokeniser(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenised["input_ids"].to(device=DEVICE)
        attention_masks = tokenised["attention_mask"].to(device=DEVICE)
        return input_ids, attention_masks
    
    def get_embeddings(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """Get embeddings for inputted texts, with batching for memory efficiency.

        Args:
            texts (list[str]): Texts to be encoded
            batch_size (int, optional): Batch size for encoding. Defaults to 64.

        Returns:
            torch.Tensor: Embeddings for inputted texts
        """

        self.eval()
        
        input_ids, attention_masks = self.tokenise(texts)
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                input_ids, attention_masks = self.tokenise(batch_texts)

                outputs = self.encoder(input_ids, attention_mask=attention_masks)
                embeddings = pool_embeddings(outputs.last_hidden_state, attention_masks)
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def aspect_sentiment_inference(self, texts: list[str], batch_size: int = 64) -> tuple[list[int], list[int]]:
        """Inference method for aspect and sentiment extraction.

        Args:
            input_ids (torch.Tensor): Token IDs tensor of shape (batch_size, seq_length).
            attention_masks (torch.Tensor): Attention mask tensor of shape (batch_size, seq_length).
            batch_size (int, optional): Batch size for inference. Defaults to 64.

        Returns:
            tuple[list[int], list[int]]: A tuple containing the list of aspect predictions and sentiment predictions.
        """

        self.eval()

        input_ids, attention_masks = self.tokenise(texts)

        aspect_predictions = []
        sentiment_predictions = []

        with torch.no_grad():
            # Run inference with attention masking in batches
            for i in tqdm(range(0, len(input_ids), batch_size), desc="Running inference"):
                batch_input_ids = input_ids[i:i+batch_size].to(DEVICE)

                # Using an attention mask to ignore padding tokens
                batch_attention_mask = attention_masks[i:i+batch_size].to(DEVICE)

                # Use forward to get logits
                aspect_logits, sentiment_logits = self.forward(batch_input_ids, batch_attention_mask)
                
                # Getting the most likely aspect and sentiment predictions
                aspect_predictions.extend(torch.argmax(aspect_logits, dim=-1).cpu().tolist())
                sentiment_predictions.extend(torch.argmax(sentiment_logits, dim=-1).cpu().tolist())

        return aspect_predictions, sentiment_predictions


class SBERTWrapper(AspectSentimentModel):
    def __init__(self, model_name: str):
        """Wrapper for Sentence-BERT to be used for comparison in revivew and release note comparison.

        Args:
            model_name (str): Name of the Sentence-BERT model to load from Hugging Face.
        """

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def aspect_sentiment_inference(self, texts: list[str], batch_size: int = 64) -> tuple[list[int], list[int]]:
        """Placeholder inference method, returns dummy aspect and sentiment predictions.
        """

        aspect_predictions = [0] * len(texts)  # Dummy aspect predictions
        sentiment_predictions = [0] * len(texts)  # Dummy sentiment predictions

        return aspect_predictions, sentiment_predictions
        

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Get embeddings for inputted texts using SBERT.

        Args:
            texts (list[str]): Texts to be encoded

        Returns:
            torch.Tensor: Embeddings for inputted texts
        """

        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)