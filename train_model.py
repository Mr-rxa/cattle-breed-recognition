import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import json
import os
import matplotlib.pyplot as plt
import config

def create_data_generators():
    """Create data generators"""
    print("🔄 Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATASET_DIR, 'train'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATASET_DIR, 'val'),
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Classes: {train_generator.num_classes}")
    print(f"📊 Train: {train_generator.samples}, Val: {val_generator.samples}")
    
    return train_generator, val_generator

def create_model(num_classes):
    """Create model with MobileNetV2"""
    print("🏗️  Building model...")
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    x = Dense(config.DENSE_UNITS, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"✅ Model created: {model.count_params():,} parameters")
    return model, base_model

def train_model():
    """Main training function"""
    print("🚀 Starting training...")
    
    # Check dataset
    if not os.path.exists(os.path.join(config.PROCESSED_DATASET_DIR, 'train')):
        print("❌ Processed dataset not found. Run split_dataset.py first!")
        return False
    
    # Create generators
    train_gen, val_gen = create_data_generators()
    
    # Create model
    model, base_model = create_model(train_gen.num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            config.BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCTION_FACTOR,
            patience=config.LR_REDUCTION_PATIENCE
        )
    ]
    
    # Phase 1: Train classifier
    print("\n🎯 Phase 1: Training classifier layers")
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
    )
    
    history1 = model.fit(
        train_gen,
        epochs=config.INITIAL_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tuning
    print("\n🔥 Phase 2: Fine-tuning")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
    )
    
    history2 = model.fit(
        train_gen,
        epochs=config.FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Load best model and save
    model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    model.save(config.MODEL_PATH)
    
    # Save class indices
    with open(config.CLASS_INDICES_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    
    # Save labels
    labels = {v: k for k, v in train_gen.class_indices.items()}
    with open(config.LABELS_PATH, 'w') as f:
        for i in range(len(labels)):
            f.write(labels[i] + '\n')
    
    # Convert to TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(config.TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model saved: {config.TFLITE_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  TFLite conversion failed: {e}")
    
    # Final evaluation
    val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)
    print(f"\n🎉 Training completed!")
    print(f"📊 Final Validation Accuracy: {val_acc:.4f}")
    print(f"📊 Final Top-3 Accuracy: {val_top3:.4f}")
    
    return True

if __name__ == "__main__":
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🖥️  Using {len(gpus)} GPU(s)")
        except:
            pass
    
    train_model()
