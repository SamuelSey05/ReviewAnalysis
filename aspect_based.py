import torch
from tqdm import tqdm
from transformers import AutoModel


class AspectSentimentExtractor(torch.nn.Module):
    def __init__(self, model_name:str,  num_aspects: int, num_sentiments: int = 3) -> None: 
        """init procedure for AspectSentimentExtractor. 

        Args:
            num_aspects (int): Number of aspects to classify.
            num_sentiments (int, optional): Number of sentiment classes. Defaults to 3.
        """
        super(AspectSentimentExtractor, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        
        # for param in self.encoder.parameters():
        #     param.requires_grad = True

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 layers of encoder parmaeters to be fine tuned
        for layer in self.encoder.transformer.layer[-4:]:
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
            embeddings (torch.Tensor): embeddings tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (torch.Tensor): attention mask tensor of shape (batch_size, seq_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the aspect logits and sentiment logits tensors.
        """

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        # Apply attention mask to the embeddings
        masked = embeddings * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)

        # Mean pooling of the masked embeddings
        mean_pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)

        x = self.dropout(mean_pooled)  # Use mean pooled representation
        aspect_logits = self.aspect_head(x)
        sentiment_logits = self.sentiment_head(x)
        return aspect_logits, sentiment_logits


    def aspect_sentiment_inference(self, input_ids: list, attention_masks: list, device: torch.device, batch_size: int = 64) -> tuple[list, list]:
        """Inference method for aspect and sentiment extraction.

        Args:
            embeddings (torch.Tensor): embeddings tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (torch.Tensor): attention mask tensor of shape (batch_size, seq_length).
            sentence_indices (list[int]): List of indices mapping sentences to their corresponding review embeddings.
            device (torch.device): Device to run the inference on.
            batch_size (int, optional): Batch size for inference. Defaults to 64.

        Returns:
            tuple[list, list]: A tuple containing the list of aspect predictions and sentiment predictions.
        """
        aspect_predictions = []
        sentiment_predictions = []

        self.eval()
        with torch.no_grad():
            # Convert to tensors if needed
            input_ids_tensor = torch.tensor(input_ids) 
            attention_masks_tensor = torch.tensor(attention_masks) 
            
            # Run inference with attention masking in batches
            for i in tqdm(range(0, len(input_ids), batch_size), desc="Running inference"):
                batch_input_ids = input_ids_tensor[i:i+batch_size].to(device)

                # Using an attention mask to ignore padding tokens
                batch_attention_mask = attention_masks_tensor[i:i+batch_size].to(device)

                # Use forward to get logits
                aspect_logits, sentiment_logits = self.forward(batch_input_ids, batch_attention_mask)
                
                # Getting the most likely aspect and sentiment predictions
                aspect_predictions.extend(torch.argmax(aspect_logits, dim=-1).cpu().tolist())
                sentiment_predictions.extend(torch.argmax(sentiment_logits, dim=-1).cpu().tolist())

        return aspect_predictions, sentiment_predictions
