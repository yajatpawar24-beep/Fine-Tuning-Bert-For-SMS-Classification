"""
Example script demonstrating how to use the trained spam classifier
"""

from inference import load_classifier
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run spam classification inference")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to classify"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load the classifier
    print(f"Loading model from {args.model_path}...")
    classifier = load_classifier(args.model_path)
    print("Model loaded successfully!\n")
    
    if args.interactive:
        # Interactive mode
        print("=== Interactive Spam Classifier ===")
        print("Type your message (or 'quit' to exit)\n")
        
        while True:
            text = input("Message: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            # Get prediction with probabilities
            prediction = classifier.predict_single(text)
            probabilities = classifier.predict_single(text, return_probabilities=True)
            
            print(f"Prediction: {prediction.upper()}")
            print(f"Confidence - Ham: {probabilities[0]:.2%}, Spam: {probabilities[1]:.2%}\n")
    
    elif args.text:
        # Single text mode
        prediction = classifier.predict_single(args.text)
        probabilities = classifier.predict_single(args.text, return_probabilities=True)
        
        print(f"Text: {args.text}")
        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence - Ham: {probabilities[0]:.2%}, Spam: {probabilities[1]:.2%}")
    
    else:
        # Demo mode with example texts
        print("=== Running demo with example texts ===\n")
        
        example_texts = [
            "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
            "Hey, are we still meeting for lunch today?",
            "URGENT: Your account has been compromised. Click this link immediately to verify.",
            "Can you pick up milk on your way home?",
            "Free entry in a weekly contest! Text WIN to 12345",
            "Meeting rescheduled to 3pm tomorrow in conference room B",
            "Get rich quick! Make $5000 per week working from home!",
            "Don't forget mom's birthday is next Tuesday"
        ]
        
        predictions = classifier.predict(example_texts, return_probabilities=False)
        probabilities = classifier.predict(example_texts, return_probabilities=True)
        
        for i, (text, pred, probs) in enumerate(zip(example_texts, predictions, probabilities), 1):
            print(f"{i}. Text: {text}")
            print(f"   Prediction: {pred.upper()}")
            print(f"   Confidence: Ham {probs[0]:.2%} | Spam {probs[1]:.2%}\n")


if __name__ == "__main__":
    main()
