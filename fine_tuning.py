import logging

import optuna
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedModel, Trainer, TrainingArguments

from datasets import Dataset, DatasetDict
from main import FINE_TUNED_MODEL_PATH

logger = logging.getLogger(__name__)

def fine_tune_model(tokenised_dataset: Dataset, model_name: str, device: torch.device, optimise_hyperparameters: bool = False) -> PreTrainedModel:
    """Fine-tune a pre-trained model on a tokenized dataset.

    Args:
        tokenised_dataset (Dataset): The tokenized dataset for training and evaluation.
        model_name (str): The name or path of the pre-trained model.
        device (torch.device): The device to run the model on.
        optimise_hyperparameters (bool, optional): Whether to perform hyperparameter optimization. Defaults to False.

    Returns:
        PreTrainedModel: The fine-tuned model.
    """
    # Split the dataset into training and testing sets
    train_test_dataset = tokenised_dataset.train_test_split(test_size=0.2)

    if optimise_hyperparameters:
        def objective(trial: optuna.trial.Trial):   
            # Hyperparameter search - tune learning rate, number of epochs, and weight decay

            # Hyperparameter suggestions to be tested
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
            num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
            weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)

            training_args = TrainingArguments(
                output_dir="./training_results",
                eval_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset = train_test_dataset["train"],
                eval_dataset= train_test_dataset["test"],
            )

            trainer.train()
            metrics = accuracy_score(train_test_dataset["test"]["sentiment"], trainer.predict(train_test_dataset["test"]).predictions.argmax(axis=1))

            return float(metrics)    
        
        logger.info("Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        logger.info("Best hyperparameters: %s", study.best_params)
        logger.info("Best accuracy: %f", study.best_value)

        # Use the best hyperparameters found
        params = study.best_params

    else:
        # Hyperparemeter values found from previous tuning
        logger.info("Using predefined hyperparameters for fine-tuning...")
        params = {
            "learning_rate": 2.13e-05,
            "num_train_epochs": 3,
            "weight_decay": 0.295,
        }

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    training_args = TrainingArguments(
        output_dir="./training_results",
        eval_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=params["num_train_epochs"],
        weight_decay=params["weight_decay"],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(FINE_TUNED_MODEL_PATH)

    return model