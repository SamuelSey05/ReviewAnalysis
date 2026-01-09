import optuna
import torch
import gc
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from datasets import Dataset


def fine_tune_model(tokenised_dataset: Dataset, model_name: str):
    def objective(trial: optuna.trial.Trial):   
        # Hyperparameter search

        learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

        training_args = TrainingArguments(
            output_dir="./training_results",
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            gradient_accumulation_steps=4,
            bf16=torch.backends.mps.is_available(), 
            dataloader_pin_memory=False,
            report_to="none",
            save_strategy="no",
            logging_steps=100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset = train_test_dataset["train"],
            eval_dataset= train_test_dataset["test"],
        )

        trainer.train()

        predictions = trainer.predict(train_test_dataset["test"])
        metrics = accuracy_score(train_test_dataset["test"]["label"], predictions.predictions.argmax(axis=1))

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return metrics
    
    print("Starting hyperparameter tuning with Optuna")

    train_test_dataset = tokenised_dataset.train_test_split(test_size=0.1)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy: ", study.best_value)

    best_params = study.best_params

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./training_results",
        eval_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        gradient_accumulation_steps=4,
        bf16=torch.backends.mps.is_available(),
        dataloader_pin_memory=False,
        report_to="none",
        save_strategy="no",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
    )

    trainer.train()

    model.save_pretrained("./models/fine_tuned_model")

    return model