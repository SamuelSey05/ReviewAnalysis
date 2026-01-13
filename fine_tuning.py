import optuna
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from datasets import Dataset


def fine_tune_model(tokenised_dataset: Dataset, model_name: str, optimise_hyperparameters: bool = False):
    train_test_dataset = tokenised_dataset.train_test_split(test_size=0.2)

    if optimise_hyperparameters:
        def objective(trial: optuna.trial.Trial):   
            # Hyperparameter search

            learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
            num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
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
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset = train_test_dataset["train"],
                eval_dataset= train_test_dataset["test"],
            )

            trainer.train()
            metrics = accuracy_score(train_test_dataset["test"]["sentiment"], trainer.predict(train_test_dataset["test"]).predictions.argmax(axis=1))

            return metrics    
        print("Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        print("Best hyperparameters: ", study.best_params)
        print("Best accuracy: ", study.best_value)

        params = study.best_params

    else:
        # Hyperparemeter values found from previous tuning
        print("Using predefined hyperparameters for fine-tuning...")
        params = {
            "learning_rate": 2.13e-05,
            "num_train_epochs": 3,
            "weight_decay": 0.295,
        }

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

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

    model.save_pretrained("./models/fine_tuned_model")

    return model