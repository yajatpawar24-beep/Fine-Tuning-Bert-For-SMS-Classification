"""
BERT Fine-tuning for SMS Spam Classification using HuggingFace Trainer
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate
import numpy as np
import torch
import argparse


def load_and_split_data(test_size=0.3, val_size=0.5, seed=42):
    """
    Load SMS spam dataset and split into train, validation, and test sets.
    
    Args:
        test_size: Proportion of data to use for validation+test
        val_size: Proportion of validation+test data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    spam_dataset = load_dataset("sms_spam")
    print(f"Loaded dataset: {spam_dataset}")
    print(f"Features: {spam_dataset['train'].features}")
    
    # Split train into train and temp (val+test)
    train_val = spam_dataset['train'].train_test_split(test_size=test_size, seed=seed)
    
    # Split temp into validation and test
    val_test = train_val['test'].train_test_split(test_size=val_size, seed=seed)
    
    train_dataset = train_val['train']
    val_dataset = val_test['train']
    test_dataset = val_test['test']
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def preprocess_data(train_dataset, val_dataset, test_dataset, tokenizer):
    """
    Tokenize the datasets using the provided tokenizer.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tuple of tokenized datasets
    """
    def preprocessing_function(example):
        return tokenizer(example['sms'], truncation=True)
    
    tokenized_train = train_dataset.map(preprocessing_function, batched=True)
    tokenized_val = val_dataset.map(preprocessing_function, batched=True)
    tokenized_test = test_dataset.map(preprocessing_function, batched=True)
    
    return tokenized_train, tokenized_val, tokenized_test


def compute_metrics(eval_preds):
    """
    Compute accuracy and F1 score for evaluation.
    
    Args:
        eval_preds: Tuple of (logits, labels)
        
    Returns:
        Dictionary with accuracy and F1 scores
    """
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"]
    }


def train_model(
    model,
    tokenized_train,
    tokenized_val,
    tokenizer,
    output_dir="./results",
    num_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01
):
    """
    Train the BERT model using HuggingFace Trainer.
    
    Args:
        model: BERT model for sequence classification
        tokenized_train: Tokenized training dataset
        tokenized_val: Tokenized validation dataset
        tokenizer: HuggingFace tokenizer
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        
    Returns:
        Trained Trainer object
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_strategy="epoch",
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        report_to=[]
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print(f"Training on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    trainer.train()
    
    return trainer


def evaluate_model(trainer, tokenized_val):
    """
    Evaluate the trained model on validation set.
    
    Args:
        trainer: Trained Trainer object
        tokenized_val: Tokenized validation dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = trainer.evaluate(eval_dataset=tokenized_val)
    print(f"Evaluation metrics: {metrics}")
    return metrics


def predict_examples(trainer, tokenizer, texts):
    """
    Make predictions on example texts.
    
    Args:
        trainer: Trained Trainer object
        tokenizer: HuggingFace tokenizer
        texts: List of text strings to classify
        
    Returns:
        List of predicted labels
    """
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = trainer.model(**inputs)
        logits = outputs.logits
    
    predictions = torch.argmax(logits, dim=-1)
    
    label_map = {0: "ham", 1: "spam"}
    pred_labels = [label_map[p.item()] for p in predictions]
    
    return pred_labels


def main(args):
    """Main training pipeline"""
    # Load model and tokenizer
    checkpoint = args.model_checkpoint
    print(f"Loading model: {checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    # Load and split data
    train_dataset, val_dataset, test_dataset = load_and_split_data(
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed
    )
    
    # Preprocess data
    tokenized_train, tokenized_val, tokenized_test = preprocess_data(
        train_dataset, val_dataset, test_dataset, tokenizer
    )
    
    # Train model
    trainer = train_model(
        model,
        tokenized_train,
        tokenized_val,
        tokenizer,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Evaluate model
    metrics = evaluate_model(trainer, tokenized_val)
    
    # Test on example texts
    example_texts = [
        "Free entry in a weekly contest. Click here to learn more!",
        "Hey, are we meeting for lunch today?"
    ]
    
    predictions = predict_examples(trainer, tokenizer, example_texts)
    
    print("\n=== Example Predictions ===")
    for text, pred in zip(example_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}\n")
    
    # Save final model
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    print(f"Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT for SMS spam classification")
    parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased",
                        help="Pre-trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for model checkpoints")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Proportion for validation+test split")
    parser.add_argument("--val-size", type=float, default=0.5,
                        help="Proportion of validation+test for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)
