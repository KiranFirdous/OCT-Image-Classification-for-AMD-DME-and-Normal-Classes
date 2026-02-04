"""
OCT Image Classification for AMD, DME, and Normal Classes using Transfer Learning
Author: Kiran Firdous
Date: July 2022
Description: This script uses transfer learning with VGG16 to classify OCT images
             into three categories: AMD, DME, and Normal.
"""

# ============================================================================
# Import Libraries
# ============================================================================
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# ============================================================================
# Configuration Parameters
# ============================================================================
class Config:
    # Image parameters
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Model parameters
    BASE_MODEL = 'VGG16'  # Options: 'VGG16', 'ResNet50', 'InceptionV3'
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 3
    CLASS_NAMES = ['AMD', 'DME', 'NORMAL']
    
    # Path parameters
    DATA_PATH = 'OCT_Dataset'  # Update this with your dataset path
    SAVE_MODEL_PATH = 'saved_models'
    LOGS_PATH = 'logs'
    
    # Augmentation parameters
    USE_AUGMENTATION = True
    AUGMENTATION_FACTOR = 2

config = Config()

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
class OCTDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = []
        self.labels = []
        self.label_dict = {'AMD': 0, 'DME': 1, 'NORMAL': 2}
        
    def load_images(self):
        """Load all OCT images from dataset folders"""
        print("Loading OCT images...")
        
        for class_name in self.label_dict.keys():
            class_path = os.path.join(self.data_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist!")
                continue
                
            image_files = [f for f in os.listdir(class_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
            
            print(f"Found {len(image_files)} images in {class_name}")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                    
                    # Resize image
                    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Normalize pixel values
                    img = img.astype('float32') / 255.0
                    
                    self.images.append(img)
                    self.labels.append(self.label_dict[class_name])
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"Total images loaded: {len(self.images)}")
        print(f"Total labels loaded: {len(self.labels)}")
        
        # Show class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {config.CLASS_NAMES[label]}: {count} images")
    
    def preprocess_data(self):
        """Prepare data for training"""
        # Convert labels to categorical
        categorical_labels = to_categorical(self.labels, num_classes=config.NUM_CLASSES)
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, categorical_labels, 
            test_size=config.TEST_SPLIT, 
            random_state=42, 
            stratify=self.labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=config.VALIDATION_SPLIT, 
            random_state=42, 
            stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train, X_val):
        """Create data generators with augmentation"""
        if config.USE_AUGMENTATION:
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )
        
        return train_generator, val_generator

# ============================================================================
# Model Creation
# ============================================================================
class OCTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        
    def create_transfer_model(self):
        """Create transfer learning model based on selected base model"""
        print(f"Creating transfer learning model with {config.BASE_MODEL}...")
        
        # Select base model
        if config.BASE_MODEL == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
            )
        elif config.BASE_MODEL == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
            )
        elif config.BASE_MODEL == 'InceptionV3':
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
            )
        else:
            raise ValueError(f"Unknown base model: {config.BASE_MODEL}")
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(config.DROPOUT_RATE/2)(x)
        predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
        
        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        optimizer = Adam(learning_rate=config.LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.model.summary()
        return self.model
    
    def train_model(self, train_generator, val_generator, steps_per_epoch, validation_steps):
        """Train the model"""
        print("Training model...")
        
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(config.SAVE_MODEL_PATH, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=config.EPOCHS,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test set"""
        print("Evaluating model...")
        
        # Predict on test set
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=config.CLASS_NAMES
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        self.plot_confusion_matrix(cm)
        
        return accuracy, y_pred_classes, y_true_classes

# ============================================================================
# Visualization Utilities
# ============================================================================
class Visualizer:
    @staticmethod
    def plot_sample_images(images, labels, num_samples=9):
        """Plot sample images from dataset"""
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(images[i])
            axes[i].set_title(config.CLASS_NAMES[np.argmax(labels[i])])
            axes[i].axis('off')
        
        plt.suptitle('Sample OCT Images from Dataset', fontsize=16)
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_training_history(history):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=config.CLASS_NAMES, 
                   yticklabels=config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# Main Execution
# ============================================================================
def main():
    # Create necessary directories
    os.makedirs(config.SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(config.LOGS_PATH, exist_ok=True)
    
    # Step 1: Load and preprocess data
    data_loader = OCTDataLoader(config.DATA_PATH)
    data_loader.load_images()
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data()
    
    # Step 2: Visualize sample images
    visualizer = Visualizer()
    visualizer.plot_sample_images(X_train, y_train)
    
    # Step 3: Create data generators
    train_generator, val_generator = data_loader.create_data_generators(X_train, X_val)
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // config.BATCH_SIZE
    validation_steps = len(X_val) // config.BATCH_SIZE
    
    # Step 4: Create and train model
    classifier = OCTClassifier()
    model = classifier.create_transfer_model()
    
    history = classifier.train_model(
        train_generator, 
        val_generator, 
        steps_per_epoch, 
        validation_steps
    )
    
    # Step 5: Visualize training history
    visualizer.plot_training_history(history)
    
    # Step 6: Evaluate model
    accuracy, y_pred, y_true = classifier.evaluate_model(X_test, y_test)
    
    # Step 7: Save final model
    model.save(os.path.join(config.SAVE_MODEL_PATH, 'final_oct_model.h5'))
    print(f"Model saved to {os.path.join(config.SAVE_MODEL_PATH, 'final_oct_model.h5')}")
    
    # Step 8: Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Base Model: {config.BASE_MODEL}")
    print(f"Total Images: {len(data_loader.images)}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
