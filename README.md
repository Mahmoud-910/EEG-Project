EEG Person Identification using CNN + RNN Hybrid Model
📋 Project Overview
This project implements a deep learning system for person identification based on EEG (electroencephalography) brainwave patterns. Using the PhysioNet EEG Motor Movement/Imagery Dataset, we train a hybrid CNN+RNN model to classify which of 109 subjects a given EEG segment belongs to.

Key Features
Hybrid Architecture: Combines CNN for spatial-frequency features and RNN for temporal dynamics
109-Class Classification: Identifies individuals based on unique brain patterns
Comprehensive Pipeline: From raw data download to model evaluation
Well-Documented: Clear, commented code suitable for academic submission
🎯 Objectives
Data Preprocessing: Load, filter, and segment EEG data from 109 subjects
Feature Extraction: Generate spectrograms and prepare temporal sequences
Model Training: Build and train a CNN+RNN hybrid architecture
Evaluation: Comprehensive analysis with confusion matrices, accuracy metrics, and visualizations
Visualization: t-SNE embeddings and per-subject performance analysis
📊 Dataset Information
PhysioNet EEG Motor Movement/Imagery Dataset
Subjects: 109 healthy volunteers
Channels: 64 EEG channels (10-10 international system)
Sampling Rate: 160 Hz
Tasks: Motor execution and motor imagery (left/right hand, both hands/feet)
Runs per Subject: 14 experimental runs
Dataset Size: ~1.5 GB
Citation:

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. 
BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. 
IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.
Download: https://physionet.org/content/eegmmidb/1.0.0/

🗂️ Project Structure
eeg_person_identification/
│
├── data/
│   ├── raw/                    # Downloaded EDF files
│   ├── processed/              # Preprocessed data (HDF5)
│   └── spectrograms/           # Generated spectrograms
│
├── notebooks/
│   ├── 01_data_download.ipynb              # Dataset download
│   ├── 02_preprocessing.ipynb              # Data preprocessing
│   ├── 03_model_training.ipynb             # Model training
│   └── 04_evaluation_visualization.ipynb   # Evaluation & analysis
│
├── src/                        # Python modules (optional)
│   ├── preprocessing.py
│   ├── model.py
│   ├── training.py
│   └── evaluation.py
│
├── models/
│   └── best_model.h5           # Trained model weights
│
├── results/
│   ├── confusion_matrix.png
│   ├── per_subject_performance.png
│   ├── tsne_visualization.png
│   ├── training_history.png
│   ├── training_history.csv
│   ├── per_subject_accuracy.csv
│   ├── performance_report.txt
│   └── model_summary.txt
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
🚀 Installation & Setup
1. Clone or Create Project Directory
bash
mkdir eeg_person_identification
cd eeg_person_identification
2. Create Virtual Environment (Recommended)
bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n eeg_project python=3.9
conda activate eeg_project
3. Install Dependencies
Create requirements.txt with the following content:

txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
mne>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
wfdb>=4.0.0
pyEDFlib>=0.1.30
tqdm>=4.62.0
h5py>=3.6.0
Install:

bash
pip install -r requirements.txt
4. Verify Installation
python
import tensorflow as tf
import mne
print(f"TensorFlow: {tf.__version__}")
print(f"MNE: {mne.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
📖 Usage Guide
Step 1: Download Dataset
Run the first notebook to download the PhysioNet dataset:

bash
jupyter notebook notebooks/01_data_download.ipynb
Options:

Automatic Download (recommended): Downloads full dataset (~1.5GB)
Manual Download: Instructions for manual download
Subset Download: Downloads 5 subjects for testing
Expected Output: ~1526 EDF files in data/raw/

Step 2: Preprocessing
Run preprocessing to filter, segment, and prepare data:

bash
jupyter notebook notebooks/02_preprocessing.ipynb
Processing Steps:

Load EDF files using MNE-Python
Apply 8-30 Hz bandpass filter (mu and beta bands)
Extract 3-second epochs around task events
Remove artifacts using amplitude thresholding
Compute spectrograms using STFT
Normalize data (z-score normalization)
Save to HDF5 format
Expected Output:

data/processed/eeg_processed_data.h5 (~2-5GB)
data/processed/preprocessing_visualization.png
Processing Time: ~30-60 minutes (depends on hardware)

Step 3: Model Training
Train the CNN+RNN hybrid model:

bash
jupyter notebook notebooks/03_model_training.ipynb
Model Architecture:

RNN Branch (Temporal Features):
├── Input: (480 timesteps, 64 channels)
├── Bidirectional LSTM (128 units)
├── Bidirectional LSTM (64 units)
├── Dense (128) + BatchNorm + Dropout
└── Dense (64) → RNN Features

CNN Branch (Spatial-Frequency Features):
├── Input: (freq, time, 64 channels)
├── Conv2D (32) + BatchNorm + MaxPool + Dropout
├── Conv2D (64) + BatchNorm + MaxPool + Dropout
├── Conv2D (128) + BatchNorm + MaxPool + Dropout
├── Global Average Pooling
├── Dense (128) + BatchNorm + Dropout
└── Dense (64) → CNN Features

Fusion & Classification:
├── Concatenate [RNN Features, CNN Features]
├── Dense (256) + BatchNorm + Dropout
├── Dense (128) + BatchNorm + Dropout
└── Dense (109, softmax) → Subject Prediction
Training Configuration:

Optimizer: Adam (lr=0.001, ReduceLROnPlateau)
Loss: Categorical Cross-Entropy
Batch Size: 32
Epochs: 100 (with early stopping)
Data Split: 70% train, 15% val, 15% test
Expected Output:

models/best_model.h5 (trained weights)
results/training_history.csv
results/training_history.png
results/model_summary.txt
Training Time: ~2-6 hours (GPU), ~12-24 hours (CPU)

Step 4: Evaluation & Visualization
Evaluate model performance and create visualizations:

bash
jupyter notebook notebooks/04_evaluation_visualization.ipynb
Evaluation Metrics:

Overall accuracy and Top-5 accuracy
Weighted F1-score, precision, recall
Macro F1-score, precision, recall
Per-subject accuracy analysis
109×109 confusion matrix
Visualizations:

Confusion matrices (full and subset)
Per-subject performance plots
t-SNE feature embeddings
Training history curves
Expected Output:

results/confusion_matrix.png
results/per_subject_performance.png
results/tsne_visualization.png
results/per_subject_accuracy.csv
results/performance_report.txt
📈 Expected Results
Performance Benchmarks
Based on similar studies on this dataset:

Metric	Expected Range	Our Goal
Overall Accuracy	70-85%	75-80%
Top-5 Accuracy	85-95%	90%+
Weighted F1-Score	0.70-0.85	0.75+
Model Comparison
Approach	Accuracy	Notes
CNN only	~65-70%	Good for spatial features
RNN only	~60-65%	Captures temporal dynamics
CNN+RNN Hybrid	75-80%	Best performance
🔬 Technical Details
Preprocessing Pipeline
Bandpass Filtering: 8-30 Hz
Mu band (8-12 Hz): Motor cortex activity
Beta band (12-30 Hz): Active thinking
Epoch Extraction: 3-second windows
Aligned to task onset (T1, T2 events)
480 samples per epoch (160 Hz × 3 sec)
Artifact Removal: Amplitude-based
Z-score threshold: ±5 standard deviations
Removes eye blinks, muscle artifacts
Spectrogram Generation: STFT
Window length: 64 samples
Overlap: 32 samples
FFT points: 128
Normalization: Z-score per channel
Mean = 0, Std = 1
Prevents channel bias
Model Components
CNN Branch:

Extracts spatial patterns across EEG channels
Identifies frequency-specific features
Uses 2D convolutions on spectrograms
RNN Branch:

Captures temporal dependencies
Bidirectional LSTM for context
Processes raw filtered signals
Fusion Layer:

Combines spatial-frequency and temporal features
Learns complementary representations
Dense layers for final classification
📊 Deliverables Checklist
For university submission, ensure you have:

 01_data_download.ipynb - Dataset download and verification
 02_preprocessing.ipynb - Complete preprocessing pipeline
 03_model_training.ipynb - Model architecture and training
 04_evaluation_visualization.ipynb - Results and analysis
 requirements.txt - All dependencies listed
 README.md - This documentation
 Confusion Matrix - 109×109 classification results
 Accuracy Metrics - Overall and per-subject
 F1-Scores - Weighted and macro averages
 Training History - Loss and accuracy curves
 Performance Report - Written analysis
 (Optional) t-SNE Visualization - Feature embeddings
🎓 Discussion Points for Report
Strengths of the Approach
Hybrid Architecture: Combines spatial and temporal features effectively
Person-Specific Patterns: Successfully identifies unique brain signatures
Robust Preprocessing: Removes artifacts and normalizes data
Cross-Session Generalization: Works across different recording sessions
Challenges & Limitations
Class Imbalance: Some subjects may have fewer clean epochs
Inter-Subject Variability: Brain patterns vary significantly across individuals
Computational Cost: Training requires significant GPU resources
Dataset Specificity: Model trained on motor imagery tasks only
Potential Improvements
Data Augmentation: Time shifting, noise injection, frequency masking
Attention Mechanisms: Focus on most discriminative channels/time points
Transfer Learning: Pre-train on larger EEG datasets
Ensemble Methods: Combine multiple models for better accuracy
Subject-Specific Fine-Tuning: Adapt model to individual subjects
Advanced Architectures: Transformers, EEGNet, DeepConvNet
🔍 Troubleshooting
Common Issues
1. Memory Error during Preprocessing

python
# Solution: Process subjects in batches
for batch in range(0, 109, 20):
    process_subjects(start=batch, end=min(batch+20, 109))
2. GPU Out of Memory

python
# Reduce batch size in config
BATCH_SIZE = 16  # Instead of 32
3. MNE Installation Issues

bash
# Try conda installation
conda install -c conda-forge mne
4. Slow Training

python
# Use mixed precision training
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
5. Low Accuracy

Check data preprocessing (filtering, normalization)
Verify epoch extraction aligns with task events
Try different hyperparameters (learning rate, dropout)
Ensure sufficient training epochs
📚 References
Dataset
Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface System. IEEE TBME.
PhysioNet EEG Motor Movement/Imagery Dataset: https://physionet.org/content/eegmmidb/1.0.0/
Related Work
Lawhern et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces.
Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization.
Craik et al. (2019). Deep learning for electroencephalogram (EEG) classification tasks: a review.
Tools & Libraries
MNE-Python: https://mne.tools/
TensorFlow: https://www.tensorflow.org/
Scikit-learn: https://scikit-learn.org/
👥 Contributors
Author: Mahmoud Ali
Course: Intelligent system
University: E-JUST\
Date: November 2025

📄 License
This project is created for educational purposes. The PhysioNet dataset is available under the Open Data Commons Attribution License v1.0.

🤝 Acknowledgments
PhysioNet for providing the EEG dataset
BCI2000 team for data collection
MNE-Python developers for excellent EEG processing tools
TensorFlow team for the deep learning framework


