import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tensorflow.keras import layers, models
import os

# Set random seeds for reproducibility
np.random.seed(456)
tf.random.set_seed(456)

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

# Check class balance
train_pos = np.sum(train_y)
train_neg = len(train_y) - train_pos
print(f"\nClass Distribution in Training:")
print(f"  Positive (toxic): {train_pos} ({train_pos/len(train_y)*100:.1f}%)")
print(f"  Negative (non-toxic): {train_neg} ({train_neg/len(train_y)*100:.1f}%)")

# Step 3: Build optimized model using best hyperparameters from tuning
d = 1024  # Input dimension (molecular fingerprints)
n_hidden = 100  # Best from tuning
learning_rate = 0.001  # Best from tuning
n_epochs = 30  # Best from tuning
batch_size = 128  # Best from tuning
dropout_rate = 0.5  # Best from tuning (50% keep probability = 0.5 dropout rate)

print("\nBuilding optimized neural network...")
print(f"  Hidden neurons: {n_hidden}")
print(f"  Dropout rate: {dropout_rate}")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {n_epochs}")
print(f"  Batch size: {batch_size}")

# Build the model
model = models.Sequential([
    # Input layer
    layers.Dense(n_hidden, activation='relu', input_shape=(d,), 
                 kernel_initializer='glorot_uniform', name='hidden_layer'),
    
    # Batch normalization for stability (new addition)
    layers.BatchNormalization(),
    
    # Dropout for regularization
    layers.Dropout(dropout_rate, name='dropout'),
    
    # Output layer
    layers.Dense(1, activation='sigmoid', 
                 kernel_initializer='glorot_uniform', name='output_layer')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']  # Add AUC as additional metric
)

# Display model summary
model.summary()

# Step 4: Add callbacks for better training
callbacks = [
    # Early stopping to prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    # Reduce learning rate when stuck
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    # TensorBoard for visualization
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs/optimized',
        histogram_freq=1
    )
]

print("\nTraining the optimized model...")

# Step 5: Train with class weights to handle imbalance
# Calculate class weights
total = len(train_y)
pos_weight = total / (2 * train_pos)
neg_weight = total / (2 * train_neg)
class_weights = {0: neg_weight, 1: pos_weight}
print(f"\nClass weights: {class_weights}")

# Train the model
history = model.fit(
    train_X, 
    train_y,
    batch_size=batch_size,
    epochs=n_epochs,
    validation_data=(valid_X, valid_y),
    callbacks=callbacks,
    class_weight=class_weights,  # Use class weights instead of sample weights
    verbose=1
)

# Step 6: Make predictions and evaluate
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

print("\nEvaluating on Validation Set...")
valid_y_pred_probs = model.predict(valid_X)
valid_y_pred = np.round(valid_y_pred_probs).flatten()

valid_accuracy = accuracy_score(valid_y, valid_y_pred)
valid_auc = roc_auc_score(valid_y, valid_y_pred_probs)
valid_f1 = f1_score(valid_y, valid_y_pred)

print(f"Validation Accuracy: {valid_accuracy:.4f}")
print(f"Validation AUC: {valid_auc:.4f}")
print(f"Validation F1 Score: {valid_f1:.4f}")

print("\nEvaluating on Test Set...")
test_y_pred_probs = model.predict(test_X)
test_y_pred = np.round(test_y_pred_probs).flatten()

test_accuracy = accuracy_score(test_y, test_y_pred)
test_auc = roc_auc_score(test_y, test_y_pred_probs)
test_f1 = f1_score(test_y, test_y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Step 7: Visualize training history
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# AUC plot (if available)
plt.subplot(1, 3, 3)
if 'auc' in history.history:
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

plt.tight_layout()
plt.savefig('optimized_model_results.png')
plt.show()

# Step 8: Save the improved model
model.save('tox21_optimized_model.h5')
print("\nOptimized model saved as 'tox21_optimized_model.h5'")

# Step 9: Comparison with original model
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print("Original model (50 neurons, 10 epochs):")
print(f"  Test Accuracy: 97.06%")
print(f"  Validation Accuracy: 96.68%")
print("\nOptimized model (100 neurons, 30 epochs):")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  Validation Accuracy: {valid_accuracy*100:.2f}%")
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test F1 Score: {test_f1:.4f}")
print("\nThe optimized model provides additional metrics (AUC, F1) for better evaluation.")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("To view TensorBoard: tensorboard --logdir=./logs/optimized")
print("Check 'optimized_model_results.png' for training visualizations")