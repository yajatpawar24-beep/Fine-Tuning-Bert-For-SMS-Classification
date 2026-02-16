"""
Utility functions for inference with trained BERT spam classifier
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SpamClassifier:
    """
    Wrapper class for BERT spam classifier inference
    """
    
    def __init__(self, model_path):
        """
        Initialize the spam classifier.
        
        Args:
            model_path: Path to the saved model directory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {0: "ham", 1: "spam"}
    
    def predict(self, texts, return_probabilities=False):
        """
        Predict spam/ham labels for input texts.
        
        Args:
            texts: Single text string or list of text strings
            return_probabilities: If True, return probabilities instead of labels
            
        Returns:
            List of predictions (labels or probabilities)
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        if return_probabilities:
            # Return softmax probabilities
            probabilities = torch.softmax(logits, dim=-1)
            return probabilities.cpu().numpy()
        else:
            # Return predicted labels
            predictions = torch.argmax(logits, dim=-1)
            pred_labels = [self.label_map[p.item()] for p in predictions]
            return pred_labels
    
    def predict_single(self, text, return_probabilities=False):
        """
        Predict spam/ham label for a single text.
        
        Args:
            text: Input text string
            return_probabilities: If True, return probabilities instead of label
            
        Returns:
            Predicted label or probability array
        """
        result = self.predict([text], return_probabilities=return_probabilities)
        return result[0]


def load_classifier(model_path):
    """
    Load a trained spam classifier from disk.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        SpamClassifier instance
    """
    return SpamClassifier(model_path)
