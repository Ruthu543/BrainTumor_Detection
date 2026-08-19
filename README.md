🧠 Brain Tumor Detection Using Deep Learning

An end-to-end Deep Learning and Computer Vision project that classifies brain MRI images into different brain tumor categories using Transfer Learning with VGG19. The trained model is integrated into a Streamlit web application, allowing users to upload an MRI image and receive a model-based prediction.

⚠️ Medical Disclaimer: This project is developed for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

📌 Project Overview

Brain tumor detection from MRI images is an important application of Computer Vision and Deep Learning. Manual analysis of medical images requires specialized expertise and can be time-consuming.

This project demonstrates how Convolutional Neural Networks (CNNs) and Transfer Learning can be used to automatically classify brain MRI images.

The project follows an end-to-end workflow:

MRI Dataset
     ↓
Exploratory Data Analysis
     ↓
Image Preprocessing
     ↓
Data Augmentation
     ↓
Transfer Learning
     ↓
VGG19 Model
     ↓
Model Training
     ↓
Fine-Tuning
     ↓
Model Evaluation
     ↓
Save Trained Model
     ↓
Streamlit Web Application
     ↓
Upload MRI Image
     ↓
Brain Tumor Prediction
🎯 Project Objectives

The main objectives of this project are:

🧠 Develop a Deep Learning model for brain tumor image classification.
🔍 Perform Exploratory Data Analysis (EDA) on MRI images.
🖼️ Preprocess MRI images for Deep Learning.
🔄 Apply data augmentation to improve model generalization.
🧠 Implement Transfer Learning using VGG19.
🔧 Experiment with fine-tuning pretrained CNN layers.
📊 Evaluate the trained model using appropriate classification metrics.
💾 Save trained models in .h5 format.
🌐 Build an interactive Streamlit web application.
📤 Allow users to upload MRI images through the application.
⚡ Generate real-time model predictions.
📦 Manage large Deep Learning model files using Git LFS.
🔬 Problem Statement

Brain tumors can vary in appearance, size, and location within MRI scans. Developing an automated image classification system can demonstrate how Deep Learning can assist in identifying patterns in medical images.

The objective of this project is to train a CNN-based classifier that learns visual features from MRI images and predicts the corresponding tumor category.

Important: The system is a machine-learning demonstration and is not intended for clinical diagnosis.

📂 Dataset

The project uses a brain MRI image dataset organized into class-specific directories.

A typical dataset structure is:

dataset/
│
├── train/
│   ├── Class_1/
│   ├── Class_2/
│   ├── Class_3/
│   └── Class_4/
│
├── test/
│   ├── Class_1/
│   ├── Class_2/
│   ├── Class_3/
│   └── Class_4/
│
└── validation/
    ├── Class_1/
    ├── Class_2/
    ├── Class_3/
    └── Class_4/

The exact class names depend on the dataset used for training.

🔄 Deep Learning Workflow
                    ┌─────────────────────┐
                    │    MRI Dataset      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       EDA           │
                    │ Class Distribution  │
                    │ Sample Visualization│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing  │
                    │ Resize & Normalize   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Augmentation    │
                    │ Rotation / Flip etc.│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Transfer Learning   │
                    │       VGG19          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Training       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fine-Tuning          │
                    │ Selected CNN Layers  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Save Best Model      │
                    │       .h5            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Web App    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Upload MRI Image     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Prediction     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Display Prediction   │
                    └─────────────────────┘
🧠 Model Architecture
VGG19 Transfer Learning

The primary Deep Learning architecture used in this project is VGG19.

VGG19 is a deep CNN architecture consisting of multiple convolutional and fully connected layers.

Instead of training the entire network from scratch, Transfer Learning is used.

Architecture
Input MRI Image
       ↓
Image Preprocessing
       ↓
VGG19 Pretrained Base
       ↓
Feature Extraction
       ↓
Global / Flattened Features
       ↓
Dense Layer
       ↓
Dropout
       ↓
Classification Layer
       ↓
Tumor Class
🔧 Transfer Learning

The pretrained VGG19 network is used as a feature extractor.

Initially, the pretrained layers can be frozen:

VGG19
 │
 ├── Convolutional Layers → Frozen
 │
 ├── Feature Extraction
 │
 └── Custom Classification Head

After initial training, selected layers can be unfrozen for fine-tuning.

VGG19
 │
 ├── Early Layers → Frozen
 │
 ├── Selected Deep Layers → Trainable
 │
 └── Classification Head

Fine-tuning allows the model to adapt pretrained visual features to MRI images.

🖼️ Image Preprocessing

MRI images are processed before being passed to the model.

The preprocessing pipeline includes:

Loading the image.
Resizing the image to the required input dimensions.
Converting the image into the required format.
Normalizing pixel values.
Expanding the image dimensions for model prediction.

Example workflow:

MRI Image
   ↓
Load Image
   ↓
Resize
   ↓
Normalize
   ↓
Convert to Tensor
   ↓
VGG19
🔄 Data Augmentation

Data augmentation is used during training to increase image diversity and reduce overfitting.

Possible augmentation techniques include:

Rotation
Horizontal flipping
Width shifting
Height shifting
Shearing
Zooming
Original MRI
      │
      ├── Rotation
      ├── Flip
      ├── Zoom
      ├── Width Shift
      └── Height Shift
             │
             ▼
       Augmented Images
🏋️ Model Training

The model is trained using the training dataset while validation data is used to monitor generalization.

The training workflow is:

Training Dataset
       ↓
Image Augmentation
       ↓
VGG19 Feature Extraction
       ↓
Classification Head
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Weight Updates
       ↓
Validation

Training parameters can be configured according to the experiment, including:

Image size
Batch size
Number of epochs
Learning rate
Optimizer
Loss function
Dropout
Class weights
📊 Model Evaluation

The model should be evaluated using multiple metrics rather than relying only on accuracy.

Metric	Description
Accuracy	Percentage of correctly classified images
Precision	Correct positive predictions relative to all positive predictions
Recall	Correctly identified positive samples
F1-Score	Balance between Precision and Recall
Confusion Matrix	Class-by-class prediction performance
📈 Performance

Add your final test results here:

Model	Accuracy	Precision	Recall	F1-Score
VGG19	XX%	XX%	XX%	XX%
Fine-Tuned VGG19	XX%	XX%	XX%	XX%

Replace the placeholder values with the actual metrics obtained from your final test dataset.

🌐 Streamlit Web Application

The trained model is integrated into a Streamlit web application.

The application provides an easy-to-use interface where a user can:

Open the Streamlit application.
Upload a brain MRI image.
Preview the uploaded image.
Pass the image through the trained model.
Receive the predicted tumor class.
View the prediction result through the web interface.
🔄 Application Prediction Workflow
User
 │
 ▼
Streamlit Interface
 │
 ▼
Upload MRI Image
 │
 ▼
Image Preview
 │
 ▼
Image Preprocessing
 │
 ▼
Load Trained VGG19 Model
 │
 ▼
Model Prediction
 │
 ▼
Prediction Probability
 │
 ▼
Predicted Tumor Class
 │
 ▼
Display Result
🛠️ Technology Stack
Category	Technologies
Programming Language	Python
Deep Learning	TensorFlow, Keras
Architecture	VGG19
Computer Vision	OpenCV, PIL
Data Processing	NumPy, Pandas
Model Evaluation	Scikit-Learn
Visualization	Matplotlib, Seaborn
Web Application	Streamlit
Notebook	Jupyter Notebook
Version Control	Git, GitHub
Large Files	Git LFS
📁 Project Structure
Brain_Tumor_Detection/
│
├── app.py
│
├── model.h5
│
├── model_weights/
│   ├── vgg19_model_01.weights.h5
│   └── vgg19_model_02.weights.h5
│
├── brain_tumor_detection.ipynb
│
├── requirements.txt
│
├── .gitignore
│
├── .gitattributes
│
└── README.md

The exact filenames may vary depending on the final version of the project.

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/Ruthu543/BrainTumor_Detection.git

Navigate to the project:

cd BrainTumor_Detection
2. Create a Virtual Environment
Windows
python -m venv venv

Activate it:

venv\Scripts\activate
macOS / Linux
python3 -m venv venv

Activate:

source venv/bin/activate
📦 3. Install Dependencies
pip install -r requirements.txt

If requirements.txt is not available, the main packages include:

pip install tensorflow streamlit opencv-python pillow numpy pandas matplotlib seaborn scikit-learn
📥 4. Git LFS

The trained .h5 models are large files and are managed using Git Large File Storage (Git LFS).

Install Git LFS:

git lfs install

Download the LFS model files:

git lfs pull

Verify:

git lfs ls-files
▶️ Running the Streamlit Application

Run:

streamlit run app.py

Streamlit will provide a local URL similar to:

http://localhost:8501

Open the URL in your browser.

🖥️ Application Usage
Step 1 — Launch Application
streamlit run app.py
Step 2 — Upload MRI Image

Use the Streamlit file uploader to select an MRI image.

Step 3 — Image Processing

The application automatically preprocesses the uploaded image.

Step 4 — Prediction

The image is passed to the trained Deep Learning model.

Step 5 — Display Result

The application displays the predicted tumor category.

Example:

Prediction: Tumor Class
Confidence: XX%
