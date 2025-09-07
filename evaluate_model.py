import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import config

def evaluate_model():
    """Evaluate trained model"""
    print("🧪 Evaluating model...")
    
    # Load model
    if os.path.exists(config.BEST_MODEL_PATH):
        model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    else:
        model = tf.keras.models.load_model(config.MODEL_PATH)
    
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATASET_DIR, 'test'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_generator)[:2]
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"📊 Test Loss: {test_loss:.4f}")
    
    # Get predictions for detailed analysis
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, output_dict=True)
    
    print("\n📈 Per-class Performance (Top 5):")
    class_f1 = [(name, metrics['f1-score']) for name, metrics in report.items() 
                if name not in ['accuracy', 'macro avg', 'micro avg', 'weighted avg']]
    class_f1.sort(key=lambda x: x, reverse=True)
    
    for name, f1 in class_f1[:5]:
        print(f"  {name}: {f1:.3f}")
    
    # Save evaluation results
    eval_results = {
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'classification_report': report
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✅ Evaluation completed!")
    return test_accuracy

if __name__ == "__main__":
    evaluate_model()
