"""
Tox21 Toxicity Prediction Neural Network
This script trains a neural network to predict compound toxicity
Using TensorFlow 2.x with Keras API
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
import os

# Set random seeds for reproducibility
np.random.seed(456)
tf.random.set_seed(456)

# Suppress warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Loading Tox21 dataset...")

# Step 1: Load the Tox21 Dataset
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

print(f"Training data shape: {train_X.shape}")
print(f"Validation data shape: {valid_X.shape}")
print(f"Test data shape: {test_X.shape}")

# Step 2: Remove extra datasets (focus on first task only)
print("\nFocusing on first toxicity task only...")
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

print(f"Training labels shape: {train_y.shape}")
print(f"Validation labels shape: {valid_y.shape}")
print(f"Test labels shape: {test_y.shape}")

# Step 3: Define model parameters
d = 1024  # Input dimension (molecular fingerprints)
n_hidden = 50  # Number of neurons in hidden layer
learning_rate = 0.001
n_epochs = 10
batch_size = 100
dropout_rate = 0.2  # 20% dropout rate (keep 80%)

print("\nBuilding neural network...")

# Step 4-7: Build the neural network with Keras
model = models.Sequential([
    # Input layer
    layers.Dense(n_hidden, activation='relu', input_shape=(d,), name='hidden_layer'),
    
    # Dropout layer for regularization
    layers.Dropout(dropout_rate, name='dropout'),
    
    # Output layer
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Step 8: Implement mini-batching training
print("\nTraining the model...")

# Create TensorBoard callback for visualization
log_dir = './logs/fcnet-tox21'
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Train the model
history = model.fit(
    train_X, 
    train_y,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_data=(valid_X, valid_y),
    callbacks=[tensorboard_callback],
    verbose=1
)

# Step 9: Make predictions and evaluate
print("\nMaking predictions on validation set...")
valid_y_pred_probs = model.predict(valid_X)
valid_y_pred = np.round(valid_y_pred_probs).flatten()

valid_accuracy = accuracy_score(valid_y, valid_y_pred)
print(f"Validation Accuracy: {valid_accuracy:.4f}")

print("\nMaking predictions on test set...")
test_y_pred_probs = model.predict(test_X)
test_y_pred = np.round(test_y_pred_probs).flatten()

test_accuracy = accuracy_score(test_y, test_y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 10: Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

print("\nModel training complete!")
print(f"Final validation accuracy: {valid_accuracy:.4f}")
print(f"Final test accuracy: {test_accuracy:.4f}")
print(f"Check the '{log_dir}' folder for TensorBoard visualizations")
print("Check 'training_metrics.png' for the loss and accuracy plots")

# Optional: Save the model
model.save('tox21_model.h5')
print("Model saved as 'tox21_model.h5'")

# Optional: Load the model later
# loaded_model = tf.keras.models.load_model('tox21_model.h5')