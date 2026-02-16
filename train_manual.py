"""
BERT Fine-tuning for SMS Spam Classification using Manual Training Loop
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
import evaluate
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
    Tokenize and prepare datasets for DataLoader.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tuple of processed datasets ready for DataLoader
    """
    def preprocessing_function(example):
        return tokenizer(example['sms'], truncation=True)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(preprocessing_function, batched=True)
    tokenized_val = val_dataset.map(preprocessing_function, batched=True)
    tokenized_test = test_dataset.map(preprocessing_function, batched=True)
    
    # Remove text column and rename label column
    tokenized_train = tokenized_train.remove_columns(['sms'])
    tokenized_val = tokenized_val.remove_columns(['sms'])
    tokenized_test = tokenized_test.remove_columns(['sms'])
    
    tokenized_train = tokenized_train.rename_column('label', 'labels')
    tokenized_val = tokenized_val.rename_column('label', 'labels')
    tokenized_test = tokenized_test.rename_column('label', 'labels')
    
    # Set format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    tokenized_test.set_format("torch")
    
    return tokenized_train, tokenized_val, tokenized_test


def create_dataloaders(tokenized_train, tokenized_val, tokenized_test, tokenizer, batch_size=8):
    """
    Create DataLoader objects for training, validation, and testing.
    
    Args:
        tokenized_train: Tokenized training dataset
        tokenized_val: Tokenized validation dataset
        tokenized_test: Tokenized test dataset
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        tokenized_train,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    val_dataloader = DataLoader(
        tokenized_val,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    test_dataloader = DataLoader(
        tokenized_test,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=3,
    learning_rate=4e-5,
    weight_decay=0.01
):
    """
    Train the model using manual training loop with Accelerate.
    
    Args:
        model: BERT model for sequence classification
        train_dataloader: Training DataLoader
        val_dataloader: Validation DataLoader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        
    Returns:
        Trained model
    """
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    print(f"Total training steps: {num_training_steps}")
    
    # Setup accelerator for distributed training
    accelerator = Accelerator()
    
    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )
    
    # Progress bar
    progress_bar = tqdm(range(num_training_steps))
    
    # Training loop
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    print("\nTraining completed!")
    return model


def evaluate_model(model, eval_dataloader):
    """
    Evaluate the model on validation or test set.
    
    Args:
        model: Trained model
        eval_dataloader: Evaluation DataLoader
        
    Returns:
        Dictionary with evaluation metrics
    """
    accelerator = Accelerator()
    
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    model.eval()
    
    eval_dataloader = accelerator.prepare(eval_dataloader)
    
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        metric_accuracy.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"])
        )
        metric_f1.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"])
        )
    
    accuracy = metric_accuracy.compute()
    f1 = metric_f1.compute()
    
    results = {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }
    
    print(f"Evaluation Results: {results}")
    return results


def main(args):
    """Main training pipeline"""
    # Load model and tokenizer
    checkpoint = args.model_checkpoint
    print(f"Loading model: {checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
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
    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        tokenized_train, tokenized_val, tokenized_test, tokenizer, batch_size=args.batch_size
    )
    
    # Verify batch structure
    for batch in train_dataloader:
        print(f"Batch structure: {[(k, v.shape) for k, v in batch.items()]}")
        break
    
    # Train model
    model = train_model(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Evaluate model
    print("\n=== Validation Results ===")
    metrics = evaluate_model(model, val_dataloader)
    
    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT for SMS spam classification (manual loop)")
    parser.add_argument("--model-checkpoint", type=str, default="bert-base-uncased",
                        help="Pre-trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./model_manual",
                        help="Output directory for saving model")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=4e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Proportion for validation+test split")
    parser.add_argument("--val-size", type=float, default=0.5,
                        help="Proportion of validation+test for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)
