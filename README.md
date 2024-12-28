# Model Training and Test Accuracy

# Test Accuracy Badge

[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)](https://github.com/your-username/mnist-classification)


## Training and Testing Logs
Here are the details of the test logs and accuracy values for the model:

- **Test Loss:** 0.028
- **Test Accuracy:** 99.4%

### Model Overview
The model is built on a Convolutional Neural Network (CNN) architecture and trained on the MNIST dataset. It uses techniques like Batch Normalization, Dropout for regularization, and the **Gradient-based Optimization Algorithm** for training.

### Model Features:
- **Batch Normalization**: Applied after each convolution layer to stabilize the learning process and speed up training.
- **Dropout**: Used to prevent overfitting by randomly setting a fraction of the input units to zero during training.
- **Accuracy Badge**: Displays the test accuracy for each run, updated automatically by GitHub Actions.

### Training Process
- **Dataset**: MNIST dataset of handwritten digits (28x28 grayscale images).
- **Training Time**: The model converged to an accuracy of over 99% in less than 20 epochs.
- **Loss Function**: Negative log-likelihood loss (`nll_loss`).
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum.

### How to Run:
1. Clone the repository.
2. Install required dependencies from `requirements.txt`.
3. Train the model by running the script `train_model.py`.
4. The test accuracy will be displayed and updated automatically in the badge.

### Additional Information
- The model training logs are automatically updated using **GitHub Actions**, which runs on every push to the main branch.
- The accuracy badge above is dynamically updated based on the latest test results.
