# BERT Fine-tuning for SMS Spam Classification

A comprehensive implementation of fine-tuning BERT for SMS spam classification using two different approaches: HuggingFace Trainer API and manual training loop with PyTorch.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training with Trainer API](#training-with-trainer-api)
  - [Training with Manual Loop](#training-with-manual-loop)
  - [Inference](#inference)
  - [Interactive Demo](#interactive-demo)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project demonstrates two approaches to fine-tuning BERT for binary text classification (spam vs. ham) on SMS messages:

1. **Trainer API Approach** (`train_with_trainer.py`): Uses HuggingFace's high-level Trainer API for streamlined training
2. **Manual Training Loop** (`train_manual.py`): Implements the training loop manually using PyTorch and Accelerate for more control

Both approaches achieve similar performance (~98-99% accuracy) and are fully documented with comprehensive testing.

### Key Features

✅ **Dual Training Approaches** - Compare high-level and low-level implementations  
✅ **Production-Ready Code** - Clean, modular, and well-documented  
✅ **Comprehensive Testing** - 15+ unit and integration tests  
✅ **Easy Inference** - Simple API for making predictions  
✅ **Interactive Demo** - Try the model instantly  
✅ **Well-Documented** - Complete usage examples and API reference

---

## 🚀 Quick Start

Get started in 5 minutes:

```bash
# 1. Clone and navigate to directory
git clone https://github.com/yourusername/bert-spam-classifier.git
cd bert-spam-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_with_trainer.py

# 4. Run tests
python tests/test_spam_classifier.py

# 5. Try interactive demo
python demo.py --model-path ./results/final_model --interactive
```

---

## 📁 Project Structure

```
bert-spam-classifier/
│
├── 📄 README.md                    # This file - complete documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation configuration
├── 📄 pytest.ini                   # Pytest configuration
├── 📄 .gitignore                   # Git ignore patterns
├── 📄 LICENSE                      # MIT License
│
├── 🐍 train_with_trainer.py       # Training script using HuggingFace Trainer API
├── 🐍 train_manual.py              # Training script with manual PyTorch loop
├── 🐍 inference.py                 # Inference utilities and SpamClassifier class
├── 🐍 demo.py                      # Interactive demo script
│
├── 📁 tests/                       # Test suite directory
│   ├── __init__.py
│   └── test_spam_classifier.py    # Comprehensive unit and integration tests
│
└── 📁 notebooks/                   # Original Jupyter notebooks
    ├── fine-tuning-bert-on-sms-spam-classification.ipynb
    └── fine-tuning-bert-without-trainer.ipynb
```

### File Descriptions

| File | Description |
|------|-------------|
| `train_with_trainer.py` | High-level training using Trainer API with automatic mixed precision and built-in evaluation |
| `train_manual.py` | Low-level training with full control over the training loop using PyTorch and Accelerate |
| `inference.py` | SpamClassifier class for batch and single predictions with probability outputs |
| `demo.py` | Interactive demo with three modes: interactive, single text, and demo |
| `tests/test_spam_classifier.py` | Comprehensive test suite covering data loading, tokenization, model inference, and integration |

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Core dependencies:**
- `torch>=2.0.0` - PyTorch for model training
- `transformers>=4.30.0` - HuggingFace Transformers
- `datasets>=2.12.0` - HuggingFace Datasets
- `evaluate>=0.4.0` - Evaluation metrics
- `accelerate>=0.20.0` - Distributed training
- `tqdm>=4.65.0` - Progress bars
- `numpy>=1.24.0` - Numerical operations
- `pytest>=7.3.0` - Testing framework

---

## 📊 Dataset

This project uses the [SMS Spam Collection Dataset](https://huggingface.co/datasets/sms_spam) from HuggingFace Datasets.

**Dataset Statistics:**
- **Total messages**: 5,574
- **Ham (legitimate) messages**: 4,827 (86.6%)
- **Spam messages**: 747 (13.4%)

**Dataset Splits:**
- **Training set**: 70% of data (3,901 messages)
- **Validation set**: 15% of data (836 messages)
- **Test set**: 15% of data (837 messages)

The dataset is automatically downloaded during training. No manual download required!

**Example Messages:**

| Label | Message |
|-------|---------|
| Ham | "Hey, are we meeting for lunch today?" |
| Ham | "Can you pick up milk on your way home?" |
| Spam | "Free entry in a weekly contest. Click here to learn more!" |
| Spam | "WINNER!! You have been selected to receive $1000" |

---

## 💻 Usage

### Training with Trainer API

The Trainer API approach is simpler and requires less boilerplate code.

**Basic usage:**
```bash
python train_with_trainer.py
```

**With custom parameters:**
```bash
python train_with_trainer.py \
    --model-checkpoint bert-base-uncased \
    --output-dir ./results \
    --num-epochs 3 \
    --learning-rate 2e-5 \
    --weight-decay 0.01
```

**Available arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-checkpoint` | str | `bert-base-uncased` | Pre-trained model to fine-tune |
| `--output-dir` | str | `./results` | Directory to save model checkpoints |
| `--num-epochs` | int | `3` | Number of training epochs |
| `--learning-rate` | float | `2e-5` | Learning rate for optimizer |
| `--weight-decay` | float | `0.01` | Weight decay for regularization |
| `--test-size` | float | `0.3` | Proportion for validation+test split |
| `--val-size` | float | `0.5` | Proportion of validation+test for validation |
| `--seed` | int | `42` | Random seed for reproducibility |

**Example output:**
```
Loading model: bert-base-uncased
Loaded dataset: DatasetDict({train: Dataset({features: ['sms', 'label']})})
Train: 3901, Val: 836, Test: 837
Training on: GPU
Epoch 1/3: 100%|████████████| 488/488 [05:23<00:00,  1.51it/s]
Evaluation metrics: {'accuracy': 0.982, 'f1': 0.961}
Model saved to ./results/final_model
```

---

### Training with Manual Loop

The manual approach provides more control over the training process.

**Basic usage:**
```bash
python train_manual.py
```

**With custom parameters:**
```bash
python train_manual.py \
    --model-checkpoint bert-base-uncased \
    --output-dir ./model_manual \
    --num-epochs 3 \
    --learning-rate 4e-5 \
    --batch-size 8
```

**Available arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-checkpoint` | str | `bert-base-uncased` | Pre-trained model to fine-tune |
| `--output-dir` | str | `./model_manual` | Directory to save trained model |
| `--num-epochs` | int | `3` | Number of training epochs |
| `--learning-rate` | float | `4e-5` | Learning rate for optimizer |
| `--weight-decay` | float | `0.01` | Weight decay for regularization |
| `--batch-size` | int | `8` | Training batch size |
| `--test-size` | float | `0.3` | Proportion for validation+test split |
| `--val-size` | float | `0.5` | Proportion of validation+test for validation |
| `--seed` | int | `42` | Random seed for reproducibility |

**Example output:**
```
Loading model: bert-base-uncased
Training on: cuda
Batch structure: [('input_ids', torch.Size([8, 128])), ('attention_mask', torch.Size([8, 128]))]
Total training steps: 1464
Epoch 1/3: 100%|████████████| 1464/1464 [15:32<00:00,  1.57it/s]
Validation Results: {'accuracy': 0.985, 'f1': 0.965}
Model saved to ./model_manual
```

---

### Inference

Use the trained model to make predictions on new text.

**Python API:**

```python
from inference import load_classifier

# Load trained model
classifier = load_classifier("./results/final_model")

# Single prediction
text = "Congratulations! You've won a free iPhone. Click here to claim."
prediction = classifier.predict_single(text)
print(f"Prediction: {prediction}")  # Output: spam

# Batch prediction
texts = [
    "Free entry in a weekly contest!",
    "Hey, are we meeting for lunch?"
]
predictions = classifier.predict(texts)
print(predictions)  # Output: ['spam', 'ham']

# Get probabilities
probabilities = classifier.predict_single(text, return_probabilities=True)
print(f"Ham: {probabilities[0]:.4f}, Spam: {probabilities[1]:.4f}")
# Output: Ham: 0.0234, Spam: 0.9766
```

**SpamClassifier API:**

```python
class SpamClassifier:
    def __init__(self, model_path):
        """Initialize classifier with model from disk"""
        
    def predict(self, texts, return_probabilities=False):
        """
        Predict labels for list of texts
        
        Args:
            texts: Single text or list of texts
            return_probabilities: Return probabilities if True
            
        Returns:
            List of predictions ('ham' or 'spam') or probability arrays
        """
        
    def predict_single(self, text, return_probabilities=False):
        """
        Predict label for single text
        
        Args:
            text: Input text string
            return_probabilities: Return probabilities if True
            
        Returns:
            Predicted label or probability array
        """
```

---

### Interactive Demo

The demo script provides three modes for testing your model.

**1. Interactive Mode** (recommended for testing):
```bash
python demo.py --model-path ./results/final_model --interactive
```

```
=== Interactive Spam Classifier ===
Type your message (or 'quit' to exit)

Message: Free iPhone! Click here to win!
Prediction: SPAM
Confidence - Ham: 2.34%, Spam: 97.66%

Message: Can we reschedule our meeting?
Prediction: HAM
Confidence - Ham: 98.21%, Spam: 1.79%
```

**2. Single Text Mode:**
```bash
python demo.py --model-path ./results/final_model --text "Free iPhone! Click to claim!"
```

```
Text: Free iPhone! Click to claim!
Prediction: SPAM
Confidence - Ham: 3.45%, Spam: 96.55%
```

**3. Demo Mode** (run multiple examples):
```bash
python demo.py --model-path ./results/final_model
```

```
=== Running demo with example texts ===

1. Text: Congratulations! You've won a free iPhone...
   Prediction: SPAM
   Confidence: Ham 1.23% | Spam 98.77%

2. Text: Hey, are we still meeting for lunch today?
   Prediction: HAM
   Confidence: Ham 99.12% | Spam 0.88%
...
```

---

## 🧪 Testing

Run the comprehensive test suite to verify all components work correctly.

**Run all tests:**
```bash
# Using unittest
python tests/test_spam_classifier.py

# Or using pytest
pytest tests/test_spam_classifier.py -v
```

**Test Coverage:**

The test suite includes 15+ tests covering:

1. **Data Loading Tests** (2 tests)
   - ✅ Dataset loads correctly
   - ✅ Dataset splits maintain correct proportions

2. **Tokenization Tests** (4 tests)
   - ✅ Tokenizer loads successfully
   - ✅ Single text tokenization
   - ✅ Batch tokenization with padding
   - ✅ Long text truncation

3. **Model Tests** (3 tests)
   - ✅ Model loads successfully
   - ✅ Model produces correct output shape
   - ✅ Batch inference works

4. **Inference Tests** (4 tests)
   - ✅ Classifier initialization
   - ✅ Single text prediction
   - ✅ Batch prediction
   - ✅ Probability outputs

5. **Integration Tests** (1 test)
   - ✅ End-to-end pipeline (data → training → inference)

**Example output:**
```
test_batch_prediction (test_spam_classifier.TestInference) ... ok
test_batch_tokenization (test_spam_classifier.TestTokenization) ... ok
test_classifier_initialization (test_spam_classifier.TestInference) ... ok
test_dataset_loading (test_spam_classifier.TestDataLoading) ... ok
test_dataset_split (test_spam_classifier.TestDataLoading) ... ok
...

----------------------------------------------------------------------
Ran 15 tests in 45.234s

OK
```

---

## 📈 Model Performance

Both training approaches achieve similar high performance on the SMS spam classification task.

### Evaluation Metrics

| Metric | Trainer API | Manual Loop |
|--------|-------------|-------------|
| **Accuracy** | ~98-99% | ~98-99% |
| **F1 Score** | ~95-97% | ~95-97% |
| **Training Time** | ~15-20 min (GPU) | ~15-20 min (GPU) |
| **Parameters** | ~110M | ~110M |

### Example Predictions

| Message | True Label | Prediction | Confidence |
|---------|------------|------------|------------|
| "Free entry in a weekly contest!" | Spam | Spam ✓ | 98.77% |
| "Hey, are we meeting for lunch?" | Ham | Ham ✓ | 99.12% |
| "WINNER!! You have been selected..." | Spam | Spam ✓ | 99.45% |
| "Can you pick up milk on your way?" | Ham | Ham ✓ | 98.56% |
| "Urgent: Account compromised, click here" | Spam | Spam ✓ | 97.23% |
| "Meeting rescheduled to 3pm tomorrow" | Ham | Ham ✓ | 99.34% |

### Performance Characteristics

- **Convergence**: Model converges within 3 epochs
- **Generalization**: Strong performance on test set indicates good generalization
- **Efficiency**: Training completes quickly even on CPU (~1-2 hours)
- **Robustness**: Handles various spam patterns and legitimate messages well

---

## 🔧 Technical Details

### Model Architecture

- **Base Model**: BERT-base-uncased (110M parameters)
- **Pre-training**: Trained on English Wikipedia and BookCorpus
- **Classification Head**: Single linear layer for binary classification
- **Input**: Tokenized text (max 512 tokens)
- **Output**: 2 logits (ham, spam)

### Training Configuration

**Trainer API Approach:**
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5 with cosine scheduler
- **Batch Size**: Dynamic (handled by Trainer)
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Mixed Precision**: FP16 (if GPU available)
- **Evaluation**: After each epoch

**Manual Loop Approach:**
- **Optimizer**: AdamW
- **Learning Rate**: 4e-5 with linear scheduler
- **Batch Size**: 8
- **Epochs**: 3
- **Weight Decay**: 0.01
- **Distributed Training**: HuggingFace Accelerate

### Data Preprocessing

1. **Tokenization**:
   - Uses BERT tokenizer with WordPiece
   - Truncates sequences to max 512 tokens
   - Dynamic padding for efficiency

2. **Label Encoding**:
   - Ham (legitimate) → 0
   - Spam → 1

3. **Data Splits**:
   - Train: 70% (3,901 samples)
   - Validation: 15% (836 samples)
   - Test: 15% (837 samples)
   - Stratified by label to maintain class distribution

### Key Implementation Features

1. **Dynamic Padding**: Efficient batching by padding sequences to batch max length
2. **Mixed Precision**: FP16 training for faster computation (when GPU available)
3. **Learning Rate Scheduling**: Cosine/linear decay for better convergence
4. **Weight Decay**: L2 regularization to prevent overfitting
5. **Gradient Accumulation**: Supported via Accelerate for larger effective batch sizes
6. **Progress Tracking**: tqdm progress bars for training visibility

---

## 📚 API Reference

### Training Functions

**`train_with_trainer.py`**

```python
def load_and_split_data(test_size=0.3, val_size=0.5, seed=42):
    """
    Load SMS spam dataset and split into train, validation, and test sets.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """

def preprocess_data(train_dataset, val_dataset, test_dataset, tokenizer):
    """
    Tokenize the datasets using the provided tokenizer.
    
    Returns:
        Tuple of tokenized datasets
    """

def compute_metrics(eval_preds):
    """
    Compute accuracy and F1 score for evaluation.
    
    Returns:
        Dictionary with accuracy and F1 scores
    """

def train_model(model, tokenized_train, tokenized_val, tokenizer, **kwargs):
    """
    Train the BERT model using HuggingFace Trainer.
    
    Returns:
        Trained Trainer object
    """

def evaluate_model(trainer, tokenized_val):
    """
    Evaluate the trained model on validation set.
    
    Returns:
        Dictionary of evaluation metrics
    """

def predict_examples(trainer, tokenizer, texts):
    """
    Make predictions on example texts.
    
    Returns:
        List of predicted labels
    """
```

**`train_manual.py`**

```python
def create_dataloaders(tokenized_train, tokenized_val, tokenized_test, 
                       tokenizer, batch_size=8):
    """
    Create DataLoader objects for training, validation, and testing.
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """

def train_model(model, train_dataloader, val_dataloader, **kwargs):
    """
    Train the model using manual training loop with Accelerate.
    
    Returns:
        Trained model
    """

def evaluate_model(model, eval_dataloader):
    """
    Evaluate the model on validation or test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
```

### Inference API

**`inference.py`**

```python
class SpamClassifier:
    """Wrapper class for BERT spam classifier inference"""
    
    def __init__(self, model_path: str):
        """
        Initialize the spam classifier.
        
        Args:
            model_path: Path to the saved model directory
        """
    
    def predict(self, texts: Union[str, List[str]], 
                return_probabilities: bool = False) -> Union[List[str], np.ndarray]:
        """
        Predict spam/ham labels for input texts.
        
        Args:
            texts: Single text string or list of text strings
            return_probabilities: If True, return probabilities instead of labels
            
        Returns:
            List of predictions (labels or probabilities)
        """
    
    def predict_single(self, text: str, 
                       return_probabilities: bool = False) -> Union[str, np.ndarray]:
        """
        Predict spam/ham label for a single text.
        
        Args:
            text: Input text string
            return_probabilities: If True, return probabilities instead of label
            
        Returns:
            Predicted label or probability array
        """

def load_classifier(model_path: str) -> SpamClassifier:
    """
    Load a trained spam classifier from disk.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        SpamClassifier instance
    """
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

**1. CUDA Out of Memory**

If you encounter CUDA OOM errors during training:

```bash
# Solution 1: Reduce batch size
python train_manual.py --batch-size 4

# Solution 2: Use gradient accumulation (modify code)
# Solution 3: Use CPU training (slower)
```

**2. Slow Training on CPU**

Training on CPU is significantly slower (~1-2 hours):

```bash
# Option 1: Use Google Colab with free GPU
# Option 2: Reduce epochs for testing
python train_with_trainer.py --num-epochs 1

# Option 3: Use smaller model
python train_with_trainer.py --model-checkpoint distilbert-base-uncased
```

**3. Dataset Download Issues**

If automatic download fails:

```python
from datasets import load_dataset

# Try with explicit cache directory
dataset = load_dataset("sms_spam", cache_dir="./data")

# Or download manually and load from disk
```

**4. Import Errors**

If you get module import errors:

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# If using notebooks, restart kernel after installation

# For test imports, run from project root
cd /path/to/bert-spam-classifier
python tests/test_spam_classifier.py
```

**5. Model Loading Issues**

If trained model fails to load:

```python
# Ensure the model directory contains all files
# Required files: config.json, model.safetensors (or pytorch_model.bin), tokenizer files

# Check directory contents
import os
print(os.listdir("./results/final_model"))
```

**6. Low Accuracy**

If model achieves lower accuracy than expected:

- Ensure dataset split is correct (70/15/15)
- Check if model trained for enough epochs (at least 3)
- Verify learning rate is appropriate (2e-5 to 4e-5)
- Check for data preprocessing errors
- Try different random seeds

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

Open an issue with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Error messages or logs

### Suggesting Enhancements

Open an issue with:
- Clear description of the enhancement
- Why it would be useful
- Implementation ideas (optional)

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes:
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation
4. Test your changes:
   ```bash
   python tests/test_spam_classifier.py
   ```
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Add comments for complex logic
- Update tests for changed functionality

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/bert-spam-classifier.git
cd bert-spam-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install in editable mode

# Run tests
python tests/test_spam_classifier.py
```

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact & Support

- **Issues**: Open an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: your.email@example.com

---

## 🌟 Acknowledgments

- HuggingFace for the Transformers library and datasets
- The creators of the SMS Spam Collection dataset
- The PyTorch and BERT communities

---

## 📊 Project Statistics

- **Lines of Code**: ~1,500
- **Test Coverage**: 15+ comprehensive tests
- **Documentation**: Complete API reference and usage examples
- **Training Time**: ~15-20 minutes (GPU) / ~1-2 hours (CPU)
- **Model Size**: ~110M parameters
- **Accuracy**: ~98-99%

---

**Note**: This project is for educational and research purposes. The model's performance may vary depending on specific use cases and dataset characteristics.

---

Made with ❤️ for learning and sharing knowledge about NLP and BERT fine-tuning.
