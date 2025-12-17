from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from transformers import TrainingArguments, Trainer

def fine_tune_model(tokenised_dataset: Dataset, model_name: str):
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_test_dataset = tokenised_dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir="./training_results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)

    model.save_pretrained("./models/fine_tuned_model")

    return model