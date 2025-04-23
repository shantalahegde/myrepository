import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gradio as gr

# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 38  # Adjust based on your dataset
DATA_DIR = "plantvillage_dataset"  # Replace with your dataset path

# Data Preparation
def prepare_data(data_dir):
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Get class names
    class_names = train_ds.class_names
    
    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

# Data Augmentation
def create_augmentation_layer():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    return data_augmentation

# Model Building (Using Transfer Learning)
def build_model(num_classes):
    # Load pre-trained EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = create_augmentation_layer()(inputs)
    
    # Preprocessing (specific to EfficientNet)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    
    # The base model contains batchnorm layers. We want to keep them in inference mode
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Fine-tuning
def fine_tune_model(model):
    # Unfreeze the top layers of the model
    model.trainable = True
    
    # Recompile the model for low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Training
def train_model(model, train_ds, val_ds, epochs=EPOCHS):
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", save_best_only=True)
    ]
    
    # Initial training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Fine-tuning
    model = fine_tune_model(model)
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs/2),
        initial_epoch=history.epoch[-1],
        callbacks=callbacks
    )
    
    return model, history, history_fine

# Evaluation
def evaluate_model(model, val_ds):
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {accuracy*100:.2f}%")
    return accuracy

# Prediction Function
def predict_image(img, model, class_names):
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    return predicted_class, confidence

# Gradio Interface
def create_interface(model, class_names):
    def classify_image(inp):
        inp = inp.astype('float32')
        predicted_class, confidence = predict_image(inp, model, class_names)
        return f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%"
    
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(),
        outputs="text",
        title="Plant Leaf Disease Detection",
        description="Upload an image of a plant leaf to detect potential diseases."
    )
    
    return interface

# Main Function
def main():
    # Prepare data
    train_ds, val_ds, class_names = prepare_data(DATA_DIR)
    print(f"Class names: {class_names}")
    
    # Build model
    model = build_model(len(class_names))
    model.summary()
    
    # Train model
    model, history, history_fine = train_model(model, train_ds, val_ds)
    
    # Evaluate model
    accuracy = evaluate_model(model, val_ds)
    
    # Save model
    model.save("plant_disease_model.h5")
    
    # Create and launch interface
    interface = create_interface(model, class_names)
    interface.launch()

if __name__ == "__main__":
    main()
