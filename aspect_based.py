import torch

class AspectExtractor(torch.nn.Module):
    def __init__(self, model_name: str, num_aspects: int, num_sentiments: int = 3):
        super(AspectExtractor, self).__init__()
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

    def forward(self, embeddings, attention_mask):
        masked = embeddings * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = masked.sum(dim=1) / lengths.unsqueeze(-1)

        x = self.dropout(mean_pooled)  # Use mean pooled representation
        aspect_logits = self.aspect_head(x)
        sentiment_logits = self.sentiment_head(x)
        return aspect_logits, sentiment_logits


    # def aspect_extraction_inference(embeddings, model_name: str, batch_size: int = 64):
    