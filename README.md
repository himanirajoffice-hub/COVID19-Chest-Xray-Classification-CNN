# COVID-19 Detection from Chest X-rays using CNN

This project uses deep learning to classify chest X-rays into COVID-19, Normal, and Viral Pneumonia.

Best Model: Fine-Tuned VGG16

Test Accuracy: 89.39%
Weighted F1 Score: 0.89

A Streamlit application is included for uploading and classifying chest X-ray images.

Disclaimer: This project is for educational purposes only and is not a medical diagnostic tool.
## Trained Model Weights

The trained Fine-Tuned VGG16 model weights are not included in this GitHub repository because the model file exceeds GitHub's standard individual file upload limit.

The complete model architecture, training process, fine-tuning procedure, evaluation metrics, and results are provided in the Jupyter Notebook.

For local deployment, the Streamlit application requires the trained `best_vgg16.weights.h5` file to be placed in the same directory as `app.py`.
