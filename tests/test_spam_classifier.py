"""
Test suite for BERT spam classifier
"""

import unittest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import SpamClassifier


class TestDataLoading(unittest.TestCase):
    """Test data loading and splitting functionality"""
    
    def test_dataset_loading(self):
        """Test that the SMS spam dataset loads correctly"""
        dataset = load_dataset("sms_spam")
        
        self.assertIn('train', dataset)
        self.assertGreater(len(dataset['train']), 0)
        self.assertIn('sms', dataset['train'].features)
        self.assertIn('label', dataset['train'].features)
    
    def test_dataset_split(self):
        """Test that dataset splits maintain correct proportions"""
        dataset = load_dataset("sms_spam")
        original_size = len(dataset['train'])
        
        train_val = dataset['train'].train_test_split(test_size=0.3, seed=42)
        val_test = train_val['test'].train_test_split(test_size=0.5, seed=42)
        
        train_size = len(train_val['train'])
        val_size = len(val_test['train'])
        test_size = len(val_test['test'])
        
        # Check splits are approximately correct
        self.assertAlmostEqual(train_size / original_size, 0.7, delta=0.01)
        self.assertAlmostEqual(val_size / original_size, 0.15, delta=0.01)
        self.assertAlmostEqual(test_size / original_size, 0.15, delta=0.01)
        
        # Check total size is preserved
        self.assertEqual(train_size + val_size + test_size, original_size)


class TestTokenization(unittest.TestCase):
    """Test tokenization functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Load tokenizer once for all tests"""
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def test_tokenizer_loads(self):
        """Test that tokenizer loads successfully"""
        self.assertIsNotNone(self.tokenizer)
    
    def test_single_text_tokenization(self):
        """Test tokenization of a single text"""
        text = "Free entry in a weekly contest!"
        tokens = self.tokenizer(text, truncation=True)
        
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)
        self.assertGreater(len(tokens['input_ids']), 0)
    
    def test_batch_tokenization(self):
        """Test tokenization of multiple texts"""
        texts = [
            "Free entry in a weekly contest!",
            "Hey, are we meeting for lunch?"
        ]
        tokens = self.tokenizer(texts, truncation=True, padding=True)
        
        self.assertEqual(len(tokens['input_ids']), 2)
        self.assertEqual(len(tokens['attention_mask']), 2)
        
        # Check that sequences are padded to same length
        self.assertEqual(len(tokens['input_ids'][0]), len(tokens['input_ids'][1]))
    
    def test_truncation(self):
        """Test that long texts are properly truncated"""
        long_text = "word " * 1000  # Very long text
        tokens = self.tokenizer(long_text, truncation=True, max_length=512)
        
        self.assertLessEqual(len(tokens['input_ids']), 512)


class TestModel(unittest.TestCase):
    """Test model loading and basic functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Load model and tokenizer once for all tests"""
        cls.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model.to(cls.device)
    
    def test_model_loads(self):
        """Test that model loads successfully"""
        self.assertIsNotNone(self.model)
    
    def test_model_output_shape(self):
        """Test that model produces correct output shape"""
        text = "Test message"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Should output logits for 2 classes
        self.assertEqual(outputs.logits.shape[-1], 2)
    
    def test_model_batch_inference(self):
        """Test model inference on batch of texts"""
        texts = [
            "Free entry in a weekly contest!",
            "Hey, are we meeting for lunch?"
        ]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Should produce logits for both texts
        self.assertEqual(outputs.logits.shape[0], 2)
        self.assertEqual(outputs.logits.shape[1], 2)


class TestInference(unittest.TestCase):
    """Test inference utilities"""
    
    def setUp(self):
        """Create a mock classifier for testing"""
        # Note: This creates an untrained model for testing structure only
        # For real tests with trained model, you would load from a checkpoint
        self.test_model_path = "bert-base-uncased"
    
    def test_classifier_initialization(self):
        """Test that SpamClassifier initializes correctly"""
        try:
            classifier = SpamClassifier(self.test_model_path)
            self.assertIsNotNone(classifier.model)
            self.assertIsNotNone(classifier.tokenizer)
        except Exception as e:
            self.fail(f"Classifier initialization failed: {e}")
    
    def test_single_prediction(self):
        """Test prediction on a single text"""
        classifier = SpamClassifier(self.test_model_path)
        text = "Free entry in a weekly contest!"
        
        prediction = classifier.predict_single(text)
        
        self.assertIn(prediction, ["ham", "spam"])
    
    def test_batch_prediction(self):
        """Test prediction on multiple texts"""
        classifier = SpamClassifier(self.test_model_path)
        texts = [
            "Free entry in a weekly contest!",
            "Hey, are we meeting for lunch?"
        ]
        
        predictions = classifier.predict(texts)
        
        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIn(pred, ["ham", "spam"])
    
    def test_probability_output(self):
        """Test that probability output works correctly"""
        classifier = SpamClassifier(self.test_model_path)
        text = "Test message"
        
        probabilities = classifier.predict_single(text, return_probabilities=True)
        
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(sum(probabilities), 1.0, delta=0.01)
        
        # Check probabilities are in valid range
        for prob in probabilities:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from data loading to prediction"""
        # Load data
        dataset = load_dataset("sms_spam")
        self.assertIsNotNone(dataset)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
        
        # Test preprocessing
        sample = dataset['train'][0]
        tokens = tokenizer(sample['sms'], truncation=True, return_tensors="pt")
        self.assertIn('input_ids', tokens)
        
        # Test inference
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        self.assertEqual(outputs.logits.shape[-1], 2)
        
        # Test prediction
        prediction = torch.argmax(outputs.logits, dim=-1)
        self.assertIn(prediction.item(), [0, 1])


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestInference))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
