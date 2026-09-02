import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(456)
tf.set_random_seed(456)

print("="*60)
print("TOX21 HYPERPARAMETER TUNING")
print("="*60)

# Step 1: Load the data once (outside the function for efficiency)
print("\nLoading Tox21 dataset...")
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Remove extra tasks (focus on first task)
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

print(f"Training data: {train_X.shape}, Validation: {valid_X.shape}, Test: {test_X.shape}")

# Step 2: Define the evaluation function
def eval_tox21_hyperparams(n_hidden=50, n_layers=1, learning_rate=0.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100,
                           weight_positives=True, random_seed=None):
    """
    Evaluate a Tox21 model with given hyperparameters.
    Returns validation accuracy.
    """
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    
    print(f"\n{'='*50}")
    print(f"Testing configuration:")
    print(f"  n_hidden: {n_hidden}")
    print(f"  n_layers: {n_layers}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  dropout_prob: {dropout_prob}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  weight_positives: {weight_positives}")
    print(f"  random_seed: {random_seed}")
    print(f"{'='*50}")
    
    d = 1024  # Input dimension
    
    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x = tf.placeholder(tf.float32, (None, d), name='x')
        y = tf.placeholder(tf.float32, (None,), name='y')
        w = tf.placeholder(tf.float32, (None,), name='w')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        # Build hidden layers
        current_input = x
        current_dim = d
        
        for layer in range(n_layers):
            with tf.name_scope(f"layer-{layer}"):
                W = tf.Variable(tf.random_normal((current_dim, n_hidden)), name=f'W_{layer}')
                b = tf.Variable(tf.random_normal((n_hidden,)), name=f'b_{layer}')
                hidden = tf.nn.relu(tf.matmul(current_input, W) + b)
                # Apply dropout
                hidden = tf.nn.dropout(hidden, keep_prob)
                current_input = hidden
                current_dim = n_hidden
        
        # Output layer
        with tf.name_scope("output"):
            W_out = tf.Variable(tf.random_normal((current_dim, 1)), name='W_out')
            b_out = tf.Variable(tf.random_normal((1,)), name='b_out')
            y_logit = tf.matmul(current_input, W_out) + b_out
            y_one_prob = tf.sigmoid(y_logit)
            y_pred = tf.round(y_one_prob)
        
        # Loss function
        with tf.name_scope("loss"):
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            
            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy
            
            l = tf.reduce_sum(entropy)
        
        # Optimizer
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
        
        # Initialize variables
        init = tf.global_variables_initializer()
    
    # Train the model
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        
        N = train_X.shape[0]
        step = 0
        
        for epoch in range(n_epochs):
            pos = 0
            epoch_loss = 0
            batch_count = 0
            
            while pos < N:
                batch_X = train_X[pos:pos+batch_size]
                batch_y = train_y[pos:pos+batch_size]
                batch_w = train_w[pos:pos+batch_size]
                
                feed_dict = {
                    x: batch_X, 
                    y: batch_y, 
                    w: batch_w, 
                    keep_prob: dropout_prob
                }
                
                _, loss = sess.run([train_op, l], feed_dict=feed_dict)
                epoch_loss += loss
                batch_count += 1
                
                step += 1
                pos += batch_size
            
            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                avg_loss = epoch_loss / batch_count
                print(f"  Epoch {epoch+1}/{n_epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Make predictions on validation set (no dropout)
        valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
    
    # Calculate validation accuracy
    valid_accuracy = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    print(f"  Validation Accuracy: {valid_accuracy:.4f}")
    
    return valid_accuracy

# Step 3: Define hyperparameter search space
print("\n" + "="*60)
print("DEFINING HYPERPARAMETER SEARCH SPACE")
print("="*60)

# Define different values to try for each hyperparameter
hidden_sizes = [50, 100, 200]           # Number of neurons per layer
num_layers = [1, 2, 3]                  # Number of hidden layers
learning_rates = [0.001, 0.0005, 0.0001] # Learning rate
dropout_probs = [0.3, 0.5, 0.7]         # Dropout keep probability
epochs_options = [30, 45, 60]           # Number of training epochs
batch_sizes = [64, 128, 256]            # Batch size
weight_options = [True, False]          # Whether to weight positive samples

# Choose a subset for initial testing
hidden_sizes = [50, 100]                # Reduced for faster testing
num_layers = [1, 2]                     # Reduced for faster testing
learning_rates = [0.001, 0.0005]        # Reduced for faster testing
dropout_probs = [0.3, 0.5]              # Reduced for faster testing
epochs_options = [30, 45]               # Reduced for faster testing
batch_sizes = [128]                     # Keep constant for now
weight_options = [True]                 # Keep constant for now

# Step 4: Set up for multiple runs (to handle random seed sensitivity)
n_runs_per_config = 3  # Number of times to repeat each configuration

# Step 5: Run hyperparameter search
print("\n" + "="*60)
print("STARTING HYPERPARAMETER SEARCH")
print(f"Testing {len(hidden_sizes) * len(num_layers) * len(learning_rates) * len(dropout_probs) * len(epochs_options)} configurations")
print(f"Each configuration will be run {n_runs_per_config} times with different seeds")
print(f"Total training runs: {len(hidden_sizes) * len(num_layers) * len(learning_rates) * len(dropout_probs) * len(epochs_options) * n_runs_per_config}")
print("="*60)

# Store results
results = []
best_accuracy = 0
best_params = None
run_id = 0

start_time = time.time()

# Nested loops for all hyperparameter combinations
for n_hidden in hidden_sizes:
    for n_layers in num_layers:
        for learning_rate in learning_rates:
            for dropout_prob in dropout_probs:
                for n_epochs in epochs_options:
                    for batch_size in batch_sizes:
                        for weight_positives in weight_options:
                            
                            # Run multiple times with different seeds
                            accuracies = []
                            for run in range(n_runs_per_config):
                                run_id += 1
                                random_seed = 456 + run_id  # Different seed for each run
                                
                                print(f"\nRun #{run_id}: Testing configuration {run+1}/{n_runs_per_config}")
                                
                                # Evaluate this configuration
                                val_acc = eval_tox21_hyperparams(
                                    n_hidden=n_hidden,
                                    n_layers=n_layers,
                                    learning_rate=learning_rate,
                                    dropout_prob=dropout_prob,
                                    n_epochs=n_epochs,
                                    batch_size=batch_size,
                                    weight_positives=weight_positives,
                                    random_seed=random_seed
                                )
                                
                                accuracies.append(val_acc)
                            
                            # Calculate average accuracy for this configuration
                            avg_accuracy = np.mean(accuracies)
                            std_accuracy = np.std(accuracies)
                            
                            print(f"\n*** Configuration Results ***")
                            print(f"  Accuracies: {accuracies}")
                            print(f"  Average: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
                            
                            # Store results
                            result = {
                                'n_hidden': n_hidden,
                                'n_layers': n_layers,
                                'learning_rate': learning_rate,
                                'dropout_prob': dropout_prob,
                                'n_epochs': n_epochs,
                                'batch_size': batch_size,
                                'weight_positives': weight_positives,
                                'accuracies': accuracies,
                                'avg_accuracy': avg_accuracy,
                                'std_accuracy': std_accuracy
                            }
                            results.append(result)
                            
                            # Track best configuration
                            if avg_accuracy > best_accuracy:
                                best_accuracy = avg_accuracy
                                best_params = result
                                print(f"  *** NEW BEST! Accuracy: {best_accuracy:.4f} ***")

# Step 6: Display results summary
print("\n" + "="*60)
print("HYPERPARAMETER SEARCH COMPLETE!")
print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
print("="*60)

# Sort results by average accuracy
results.sort(key=lambda x: x['avg_accuracy'], reverse=True)

print("\nTOP 5 CONFIGURATIONS:")
print("-" * 60)
for i, result in enumerate(results[:5]):
    print(f"\n#{i+1}: Average Accuracy: {result['avg_accuracy']:.4f} (+/- {result['std_accuracy']:.4f})")
    print(f"  n_hidden: {result['n_hidden']}")
    print(f"  n_layers: {result['n_layers']}")
    print(f"  learning_rate: {result['learning_rate']}")
    print(f"  dropout_prob: {result['dropout_prob']}")
    print(f"  n_epochs: {result['n_epochs']}")
    print(f"  batch_size: {result['batch_size']}")
    print(f"  weight_positives: {result['weight_positives']}")
    print(f"  Individual accuracies: {result['accuracies']}")

print("\n" + "="*60)
print("BEST CONFIGURATION:")
print("="*60)
print(f"Average Validation Accuracy: {best_params['avg_accuracy']:.4f}")
print(f"Configuration:")
for key, value in best_params.items():
    if key not in ['accuracies', 'avg_accuracy', 'std_accuracy']:
        print(f"  {key}: {value}")

# Step 7: Visualize results
print("\nGenerating visualization...")

# Extract data for plotting
config_names = [f"H{res['n_hidden']}_L{res['n_layers']}_LR{res['learning_rate']}_D{res['dropout_prob']}_E{res['n_epochs']}" 
                for res in results[:10]]
accuracies = [res['avg_accuracy'] for res in results[:10]]
std_devs = [res['std_accuracy'] for res in results[:10]]

plt.figure(figsize=(12, 6))
plt.bar(range(len(config_names)), accuracies, yerr=std_devs, capsize=5)
plt.xlabel('Configuration')
plt.ylabel('Validation Accuracy')
plt.title('Top 10 Hyperparameter Configurations (with Error Bars)')
plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right')
plt.ylim(0.85, 1.0)
plt.tight_layout()
plt.savefig('hyperparameter_tuning_results.png')
print("Results visualization saved as 'hyperparameter_tuning_results.png'")

# Step 8: Save results to file
with open('hyperparameter_results.txt', 'w') as f:
    f.write("TOX21 HYPERPARAMETER TUNING RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total configurations tested: {len(results)}\n")
    f.write(f"Runs per configuration: {n_runs_per_config}\n")
    f.write(f"Best accuracy: {best_params['avg_accuracy']:.4f}\n\n")
    f.write("Best Configuration:\n")
    for key, value in best_params.items():
        if key not in ['accuracies', 'avg_accuracy', 'std_accuracy']:
            f.write(f"  {key}: {value}\n")
    f.write(f"\nIndividual accuracies: {best_params['accuracies']}\n\n")
    f.write("All Results (sorted by accuracy):\n")
    f.write("-"*60 + "\n")
    for i, result in enumerate(results):
        f.write(f"\n{i+1}. Accuracy: {result['avg_accuracy']:.4f}\n")
        f.write(f"   Hidden: {result['n_hidden']}, Layers: {result['n_layers']}, LR: {result['learning_rate']}\n")
        f.write(f"   Dropout: {result['dropout_prob']}, Epochs: {result['n_epochs']}, Batch: {result['batch_size']}\n")

print("\nResults saved to 'hyperparameter_results.txt'")
print("\nTuning complete! Use the best configuration in your main model.")