import torch

FINE_TUNED_MODEL_PATH = "./models/fine_tuned_model"
DISTILBERT_BASE = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
POSITIVE_SENTIMENT_THRESHOLD = 7.5
NEUTRAL_SENTIMENT_THRESHOLD = 4.0
NUM_EPOCHS = 5
DATASET_PATH = "datasets/AWARE_Comprehensive.csv"
OVERLAP_KEYWORDS_FOR_RELEASE_FILTERING = {
	"crash",
	"freeze",
	"login",
	"signin",
	"sign",
	"authenticate",
	"password",
	"notification",
	"reaction",
	"button",
	"upload",
	"download",
	"search",
	"message",
	"unread",
	"channel",
	"draft",
	"connect",
	"sync",
	"scroll",
	"share",
	"reply",
	"delete",
	"thread",
	"threads",
	"call",
	"calls",
	"audio",
	"speaker",
	"bluetooth",
	"video",
	"attachment",
	"workspace",
	"emoji",
	"huddle",
	"install",
	"reminder",
	"schedule",
	"copy",
	"paste",
	"file",
	"files",
	"photo",
	"photos",
}