# optuna_transformer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import optuna

def optimize_transformer_with_optuna(train_df, test_df, model_name='pdelobelle/robbert-2023-dutch-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    def tokenize_func(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)
    
    train_tokenized = train_dataset.map(tokenize_func, batched=True)
    test_tokenized = test_dataset.map(tokenize_func, batched=True)
    
    # Define a model initialization function rather than a static model.
    # The Trainer needs this to re-instantiate the model at the start of each trial.
    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # We set baseline training arguments. Some of these will be overridden by the Optuna search.
    training_args = TrainingArguments(
        output_dir="./optuna_transformer_results",
        evaluation_strategy="epoch",
        save_strategy="no",             # Avoid clogging disk space with intermediate trial weights
        disable_tqdm=True,              # Keeps console output clean during optimization
        logging_steps=50,
        weight_decay=0.01,
        fp16=True                       # Run in half-precision to speed up optimization on GPU
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # Define hyperparameter space for the search
    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
        }
    
    print("Launching Optuna search for Hugging Face Trainer...")
    # The search automatically integrates Optuna's median pruner to cut off unpromising trials
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=my_hp_space,
        n_trials=10, # Adjust depending on compute budget
        compute_objective=lambda metrics: metrics["eval_accuracy"]
    )
    
    print("\n--- Best Transformer Hyperparameters ---")
    print(best_run.hyperparameters)
    return best_run.hyperparameters