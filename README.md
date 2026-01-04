# Emoji Recommendation based on facial expression using CNN

## Overview
This project, conducted under the mentorship of SURE Trust from December 2024 to April 2025, focuses on developing a deep learning model to classify seven human emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) from facial images. The goal was to achieve a better model, which was met with a final custom Convolutional Neural Network (CNN) achieving **78.46% validation accuracy**. This work demonstrates practical skills in AI and computer vision, aligning with SURE Trust's mission to enhance employability for rural youth.

## Project Details
- **Domain**: Artificial Intelligence and Computer Vision
- **Duration**: December 2024 to April 2025
- **Contributor**: R. Darwin Hreesh
- **Mentors**: Gaurav Patel and Prof. Radhakumari (Executive Director & Founder, SURE Trust)
- **Dataset**: Approximately 33,404 augmented samples (e.g., sourced from RAF-DB or custom dataset)
- **Target Accuracy**: 70-80%
- **Achieved Accuracy**: 78.46% (validation)

## Features
- **Emotion Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Architectures Explored**:
  - Pre-trained MobileNetV2 (initial accuracy ~60-61%)
  - Custom CNN (final accuracy 78.46%)
- **Data Augmentation**: Rotations (30-40°), zooms (0.2-0.3), horizontal flips, brightness adjustments (0.7-1.3)
- **Optimization**:
  - Dropout (0.15-0.5), L2 regularization (0.0005-0.01)
  - Early stopping (patience 30), learning rate reduction
- **Evaluation**: Accuracy/loss plots, confusion matrix

## Project Structure
- `train_custom_cnn.py`: Main script for training the custom CNN
- `data_augmentation.py`: Script for data preprocessing and augmentation
- `visualize_results.py`: Code to generate accuracy/loss plots and confusion matrix
- `models/`: Directory containing saved model weights
- `results/`: Directory with plots and screenshots (e.g., accuracy plot, confusion matrix)

## Requirements
To run this project, you’ll need the following dependencies:
- Python 3.8+
- TensorFlow 2.5+
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install them using:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Setup and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/emotion-classification.git
   cd emotion-classification
   ```

2. **Prepare the Dataset**:
   - Place your dataset in a directory (e.g., `data/`) with subfolders for each emotion class (`Angry/`, `Disgust/`, etc.).
   - Update `train_custom_cnn.py` with the path to your dataset (`train_dir` and `test_dir`).

3. **Run Data Augmentation**:
   ```bash
   python data_augmentation.py
   ```
   This script augments the dataset to ~33,404 samples using ImageDataGenerator.

4. **Train the Model**:
   ```bash
   python train_custom_cnn.py
   ```
   - The script trains the custom CNN for 50 epochs.
   - Model weights are saved in `models/` via ModelCheckpoint.

5. **Visualize Results**:
   ```bash
   python visualize_results.py
   ```
   - Generates accuracy/loss plots and confusion matrix, saved in `results/`.

## Results
- **Initial Model (MobileNetV2)**: ~60-61% validation accuracy
- **Final Model (Custom CNN)**: **78.46% validation accuracy** (Epoch 26)
- **Training Accuracy**: 56.87% (Epoch 43)
- **Challenges Addressed**:
  - Overfitting mitigated with Dropout and regularization
  - Underfitting tackled by increasing model capacity (e.g., Dense units 1024, 512)

### Screenshots
- **Accuracy and Loss Plot**:  
  ![Accuracy and Loss](Mobilenet%20Accuracy.png)  
- **Confusion Matrix**:  
  ![Confusion Matrix](results/confusion_matrix.png)  
- **Augmented Images**:  
  ![Augmented Images](results/augmented_images.png)

## Future Improvements
- Implement a separate test set for unbiased evaluation.
- Test 224x224 input size to push accuracy beyond 80%.
- Address the train-validation gap (56.87% vs. 78.46%) with more data or cross-validation.
- Explore advanced architectures like ResNet or ensemble methods.

## Acknowledgments
- SURE Trust for providing this opportunity to gain industry-relevant skills.
- Mentors Gaurav Patil, and Prof. Radhakumari for guidance.
- [Optional: Add dataset source or other acknowledgments]


