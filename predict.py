import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image
import config

class CattleBreedPredictor:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and class names"""
        try:
            # Load model
            if os.path.exists(config.BEST_MODEL_PATH):
                self.model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
            else:
                self.model = tf.keras.models.load_model(config.MODEL_PATH)
            
            # Load class names
            if os.path.exists(config.CLASS_INDICES_PATH):
                with open(config.CLASS_INDICES_PATH, 'r') as f:
                    class_indices = json.load(f)
                self.class_names = {v: k for k, v in class_indices.items()}
                self.class_names = [self.class_names[i] for i in range(len(self.class_names))]
            else:
                with open(config.LABELS_PATH, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            
            print(f"✅ Model loaded with {len(self.class_names)} classes")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            
            return x
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def predict(self, img_path, top_k=5):
        """Predict breed from image"""
        try:
            # Preprocess
            processed_img = self.preprocess_image(img_path)
            
            # Predict
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get top K predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'breed': self.class_names[idx],
                    'confidence': float(predictions[idx]),
                    'confidence_percent': float(predictions[idx] * 100)
                })
            
            return results
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

def main():
    """Test prediction"""
    predictor = CattleBreedPredictor()
    
    # Test with an image
    test_image = "test_image.jpg"  # Replace with your test image path
    
    if os.path.exists(test_image):
        results = predictor.predict(test_image)
        
        print(f"\n📸 Predictions for: {test_image}")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            breed = result['breed']
            confidence = result['confidence_percent']
            print(f"{i}. {breed}: {confidence:.2f}%")
    else:
        print(f"❌ Test image not found: {test_image}")
        print("Available classes:")
        for i, breed in enumerate(predictor.class_names, 1):
            print(f"{i}. {breed}")

if __name__ == "__main__":
    main()
