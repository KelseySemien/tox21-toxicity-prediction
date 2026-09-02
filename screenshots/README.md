# Tox21 Toxicity Prediction with Neural Networks

A deep learning project for predicting compound toxicity using the Tox21 dataset, featuring hyperparameter tuning, TensorBoard visualization, and model optimization.

## 📋 Project Overview

This project uses the Tox21 dataset (a collection of 12,000 compounds tested for toxicity across 12 different assays) to build neural network models that predict whether a given compound is toxic. The project demonstrates:

- Building neural networks with TensorFlow 2.x and DeepChem
- Systematic hyperparameter tuning (hidden layers, neurons, learning rate, dropout, epochs)
- TensorBoard integration for monitoring training
- Model optimization with class weighting and batch normalization
- Comprehensive evaluation using accuracy, AUC, and F1 score

## 📁 Files in This Repository

| File | Description |
|------|-------------|
| `tox21_model.py` | Original Keras/TF 2.x implementation with baseline architecture |
| `tox21_hyperparameter_tuning.py` | Full hyperparameter search with 32 configurations (3 runs each) |
| `tox21_optimized_model.py` | Final optimized model using best hyperparameters from tuning |
| `requirements.txt` | All Python dependencies with version numbers |
| `hyperparameter_results.txt` | Complete tuning results with all configurations |
| `hyperparameter_tuning_results.png` | Bar chart visualization of top 10 configurations |
| `training_metrics.png` | Loss and accuracy plots from the original model |
| `optimized_model_results.png` | Training curves for the optimized model |
| `Screenshot_*.png` | TensorBoard visualizations showing loss curves and model graphs |

## 🚀 How to Run

### Prerequisites
- Python 3.7+
- TensorFlow 2.10+
- DeepChem 2.7.1
- scikit-learn 1.0.2
- Matplotlib 3.5.3
- NumPy 1.21.6

### Installation
```bash
pip install -r requirements.txt

## Run the Origional Model
python tox21_model.py
## Run Hyperparameter Tuning
python tox21_hyperparameter_tuning.py
## Run Optimized Model
python tox21_optimized_model.py
## View TensorBoard 
tensorboard --logdir=./logs


🏗️ Model Architecture
Baseline Model
Input: 1024 molecular fingerprints

Hidden Layer: 50 neurons with ReLU activation

Dropout: 20% (keep 80%)

Output Layer: 1 neuron with sigmoid activation

Loss: Binary Cross-Entropy

Optimizer: Adam (learning rate 0.001)

Epochs: 10

Batch Size: 100

Optimized Model (After Tuning)
Input: 1024 molecular fingerprints

Hidden Layer: 100 neurons with ReLU activation

Batch Normalization for training stability

Dropout: 50% (keep 50%)

Output Layer: 1 neuron with sigmoid activation

Loss: Binary Cross-Entropy with class weighting

Optimizer: Adam (learning rate 0.001)

Epochs: 30 (with early stopping)

Batch Size: 128

📊 Results
Baseline Model
Metric	Validation	Test
Accuracy	96.68%	97.06%
Optimized Model
Metric	Validation	Test
Accuracy	~90%	~90%
AUC	~0.95	~0.95
F1 Score	~0.89	~0.89
Note: The optimized model shows slightly lower accuracy but includes more meaningful metrics (AUC, F1) and uses class weighting to handle imbalanced data. The accuracy difference reflects more realistic performance measurement.

🔬 Hyperparameter Tuning
Search Space (32 Configurations)
Hyperparameter	Values Tested
Hidden Neurons	50, 100
Hidden Layers	1, 2
Learning Rate	0.001, 0.0005
Dropout Rate	0.3, 0.5
Epochs	30, 45
Batch Size	128
Best Configuration
Hidden Neurons: 100

Hidden Layers: 1

Learning Rate: 0.001

Dropout: 0.5

Epochs: 30

Batch Size: 128

Validation Accuracy: 0.7046 (average of 3 runs)

Why This Range of Accuracies?
Class Imbalance: Tox21 dataset is imbalanced (fewer toxic compounds)

Weighted Accuracy: The tuning script uses weighted accuracy based on sample weights

Random Seed Sensitivity: Neural networks are sensitive to initialization, which is why 3 runs per config were averaged

Different Metrics: The optimized model reports AUC and F1 for a complete picture

📝 Assignment Context
This project was completed for a graduate-level machine learning capstone course (CSC580) with two main objectives:

Module 4: Build a baseline neural network for toxicity prediction using TensorFlow and DeepChem

Module 5: Improve model performance through systematic hyperparameter tuning and optimization

🎓 Key Learnings
Hyperparameter Tuning: The search space is critical—too small misses optimal configs, too large is computationally expensive

Random Seed Sensitivity: Always run multiple trials per configuration

Evaluation Metrics: Accuracy alone is misleading for imbalanced data—use AUC, F1, precision, recall

Class Weighting: Essential for imbalanced datasets like Tox21

TensorBoard: Invaluable for debugging and monitoring training

Dropout vs Regularization: Dropout helps prevent overfitting but must be tuned carefully

🛠️ Technologies Used
TensorFlow 2.x / TensorFlow 1.x (compat mode)

DeepChem 2.7.1 (molecular data loading)

scikit-learn (accuracy metrics, RandomForest baseline)

Matplotlib (visualization)

NumPy (numerical computations)

TensorBoard (training visualization)

🔮 Future Improvements
Try deeper architectures (3+ hidden layers)

Experiment with different activation functions (ELU, LeakyReLU)

Use weight decay (L2 regularization) in addition to dropout

Implement learning rate scheduling (cosine annealing)

Try other molecular representations (molecular graphs, SMILES)

Use ensemble methods combining multiple models