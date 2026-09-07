# 🧠 Brain Tumor Detection Using Deep Learning

> **Deep Learning • Computer Vision • Transfer Learning • VGG19 • Streamlit**

An end-to-end **Deep Learning and Computer Vision application** that classifies brain MRI images into different tumor categories using **Transfer Learning with VGG19**.

The trained model is integrated with a **Streamlit web application**, allowing users to upload an MRI image and receive a model-generated prediction through an interactive interface.

> ⚠️ **Medical Disclaimer:** This project is intended strictly for educational and research purposes. It is **not a clinically validated diagnostic system** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 📌 Overview

Brain tumor classification is a challenging Computer Vision problem because MRI images can contain variations in tumor appearance, size, shape, and location.

This project demonstrates how **Convolutional Neural Networks (CNNs)** and **Transfer Learning** can be applied to MRI image classification.

The project covers the complete machine-learning pipeline:

```text
MRI Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Image Preprocessing
     │
     ▼
Data Augmentation
     │
     ▼
Transfer Learning
     │
     ▼
VGG19 Model
     │
     ▼
Model Training
     │
     ▼
Fine-Tuning
     │
     ▼
Model Evaluation
     │
     ▼
Trained Model
     │
     ▼
Streamlit Application
     │
     ▼
MRI Upload → Prediction → Result
```

---

## 🎯 Objectives

* Develop a Deep Learning model for brain MRI classification
* Perform Exploratory Data Analysis on MRI images
* Preprocess and prepare images for model training
* Apply data augmentation to improve model generalization
* Implement Transfer Learning using **VGG19**
* Experiment with CNN fine-tuning
* Evaluate the model using classification metrics
* Save trained models in `.h5` format
* Build an interactive Streamlit application
* Enable users to upload MRI images for prediction
* Manage large model files using Git LFS

---

## 🚀 Key Features

| Feature              | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| 🧠 Deep Learning     | CNN-based MRI image classification                         |
| 🔄 Transfer Learning | Uses pretrained VGG19                                      |
| 🔧 Fine-Tuning       | Selected pretrained layers can be unfrozen                 |
| 🖼️ Image Processing | Resize, normalize and prepare MRI images                   |
| 🔄 Data Augmentation | Rotation, flipping, shifting, zooming and shearing         |
| 📊 Model Evaluation  | Accuracy, Precision, Recall, F1-Score and Confusion Matrix |
| 🌐 Web Application   | Interactive Streamlit interface                            |
| 📤 Image Upload      | Upload MRI images directly through the application         |
| ⚡ Prediction         | Generate model-based predictions                           |
| 📦 Git LFS           | Manage large `.h5` model files                             |

---

## 🗂️ Dataset

The project uses a brain MRI image dataset organized into class-specific directories.

Recommended structure:

```text
dataset/
│
├── train/
│   ├── Class_1/
│   ├── Class_2/
│   ├── Class_3/
│   └── Class_4/
│
├── validation/
│   ├── Class_1/
│   ├── Class_2/
│   ├── Class_3/
│   └── Class_4/
│
└── test/
    ├── Class_1/
    ├── Class_2/
    ├── Class_3/
    └── Class_4/
```

The exact class names depend on the dataset used during training.

---

# 🧠 Model Architecture

## VGG19 Transfer Learning

The primary architecture used in this project is **VGG19**, a deep Convolutional Neural Network commonly used for image classification and feature extraction.

Instead of training the complete network from scratch, the project uses **Transfer Learning** to leverage pretrained visual features.

### Architecture

```text
Input MRI Image
       │
       ▼
Image Preprocessing
       │
       ▼
Pretrained VGG19
       │
       ▼
Feature Extraction
       │
       ▼
Classification Head
       │
       ▼
Dropout
       │
       ▼
Output Layer
       │
       ▼
Predicted Tumor Class
```

---

## 🔄 Transfer Learning & Fine-Tuning

The initial training stage uses the pretrained VGG19 network as a feature extractor.

```text
VGG19
 │
 ├── Early Convolutional Layers
 │          ↓
 │       Frozen
 │
 ├── Deep Feature Extraction
 │
 └── Custom Classification Head
```

After the initial training phase, selected deeper layers can be unfrozen and trained with the classification head.

```text
VGG19
 │
 ├── Early Layers
 │       ↓
 │    Frozen
 │
 ├── Selected Deep Layers
 │       ↓
 │   Trainable
 │
 └── Classification Head
```

Fine-tuning allows the pretrained network to adapt its learned visual representations to the characteristics of MRI images.

---

# 🖼️ Image Preprocessing

Before an MRI image is passed to the model, it goes through a preprocessing pipeline.

### Processing steps

1. Load the MRI image
2. Resize the image to the required dimensions
3. Convert the image into the required format
4. Normalize pixel values
5. Expand dimensions for model prediction
6. Pass the processed image to VGG19

```text
MRI Image
    │
    ▼
Load Image
    │
    ▼
Resize
    │
    ▼
Normalize
    │
    ▼
Convert to Tensor
    │
    ▼
VGG19
    │
    ▼
Prediction
```

---

# 🔄 Data Augmentation

Data augmentation is applied during training to increase image diversity and help reduce overfitting.

The augmentation pipeline can include:

* Rotation
* Horizontal flipping
* Width shifting
* Height shifting
* Shearing
* Zooming

```text
                Original MRI
                     │
       ┌─────────────┼─────────────┐
       │             │             │
    Rotation       Flip          Zoom
       │             │             │
       ├─────── Shifting ──────────┤
                     │
                     ▼
              Augmented Images
```

---

# 🏋️ Model Training

The model is trained using the training dataset while validation data is used to monitor generalization.

```text
Training Dataset
       │
       ▼
Data Augmentation
       │
       ▼
VGG19 Feature Extraction
       │
       ▼
Classification Head
       │
       ▼
Loss Calculation
       │
       ▼
Backpropagation
       │
       ▼
Weight Updates
       │
       ▼
Validation
```

Training parameters can be configured according to the experiment, including:

* Image size
* Batch size
* Number of epochs
* Learning rate
* Optimizer
* Loss function
* Dropout
* Class weights

---

# 📊 Model Evaluation

Model performance should be evaluated using multiple classification metrics rather than relying only on accuracy.

| Metric               | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| **Accuracy**         | Percentage of correctly classified images                  |
| **Precision**        | Proportion of predicted positive samples that are correct  |
| **Recall**           | Proportion of actual positive samples correctly identified |
| **F1-Score**         | Harmonic mean of precision and recall                      |
| **Confusion Matrix** | Shows class-wise prediction performance                    |

### Performance

Add the final test-set metrics obtained from your trained model here.

| Model            | Accuracy | Precision | Recall | F1-Score |
| ---------------- | -------: | --------: | -----: | -------: |
| VGG19            |        — |         — |      — |        — |
| Fine-Tuned VGG19 |        — |         — |      — |        — |

> **Note:** Replace the values above with the actual results from your final test evaluation. Avoid publishing placeholder percentages.

---

# 🌐 Streamlit Web Application

The trained model is integrated into a **Streamlit web application** that provides a simple interface for image-based prediction.

### Application capabilities

* Upload a brain MRI image
* Preview the uploaded image
* Preprocess the image automatically
* Load the trained Deep Learning model
* Generate a prediction
* Display the predicted tumor category
* Display prediction confidence when available

### Prediction Pipeline

```text
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
Load Trained Model
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
```

---

# 🛠️ Technology Stack

| Category                  | Technologies        |
| ------------------------- | ------------------- |
| **Programming Language**  | Python              |
| **Deep Learning**         | TensorFlow, Keras   |
| **Model Architecture**    | VGG19               |
| **Computer Vision**       | OpenCV, PIL         |
| **Data Processing**       | NumPy, Pandas       |
| **Model Evaluation**      | Scikit-learn        |
| **Visualization**         | Matplotlib, Seaborn |
| **Web Framework**         | Streamlit           |
| **Development**           | Jupyter Notebook    |
| **Version Control**       | Git, GitHub         |
| **Large File Management** | Git LFS             |

---

# 📁 Project Structure

```text
Brain_Tumor_Detection/
│
├── app.py
├── brain_tumor_detection.ipynb
│
├── model.h5
│
├── model_weights/
│   ├── vgg19_model_01.weights.h5
│   └── vgg19_model_02.weights.h5
│
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

> The exact filenames may vary depending on the final version of the project.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Ruthu543/BrainTumor_Detection.git
```

Navigate to the project directory:

```bash
cd BrainTumor_Detection
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install tensorflow streamlit opencv-python pillow numpy pandas matplotlib seaborn scikit-learn
```

---

# 📦 Git LFS Setup

The trained `.h5` model files can be large, so **Git Large File Storage (Git LFS)** is used to manage them.

Install and initialize Git LFS:

```bash
git lfs install
```

Download the model files:

```bash
git lfs pull
```

Verify tracked LFS files:

```bash
git lfs ls-files
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🖥️ How to Use

### Step 1 — Start the Application

```bash
streamlit run app.py
```

### Step 2 — Upload an MRI Image

Use the file uploader to select a brain MRI image.

### Step 3 — Image Processing

The application automatically preprocesses the uploaded image.

### Step 4 — Generate Prediction

The processed image is passed to the trained Deep Learning model.

### Step 5 — View Result

The application displays the predicted tumor category and, where implemented, the prediction confidence.

Example:

```text
Prediction: Tumor Class
Confidence: XX%
```

---

# 📈 Future Improvements

Potential improvements for future versions include:

* Compare VGG19 with modern architectures such as EfficientNet and ResNet
* Improve model performance through advanced fine-tuning
* Add Grad-CAM for model interpretability
* Add prediction history
* Improve the Streamlit user interface
* Add model performance visualizations
* Deploy the application to a cloud platform
* Add automated model monitoring
* Expand the dataset with more diverse MRI images

---

# 🔬 Project Highlights

This project demonstrates practical experience with:

* Deep Learning
* Convolutional Neural Networks
* Transfer Learning
* VGG19
* Fine-Tuning
* Computer Vision
* Image Preprocessing
* Data Augmentation
* Model Evaluation
* Streamlit Application Development
* Git & GitHub
* Git LFS

---

# ⚠️ Disclaimer

This application is a **Deep Learning demonstration for educational and research purposes**.

The predictions generated by this system should **not be considered medical advice or a clinical diagnosis**. Always consult a qualified healthcare professional for medical evaluation and diagnosis.

---

# 👩‍💻 Author

**Ruthu Madhavi**

GitHub:
https://github.com/Ruthu543

---

## ⭐ If You Find This Project Useful

If this project helped you learn about Deep Learning, Computer Vision, or Transfer Learning, consider giving the repository a ⭐ on GitHub.
