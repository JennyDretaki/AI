# Artificial Intelligence Challenges

This repository contains two Artificial Intelligence projects developed for the **Artificial Intelligence course at Thomas More**:

1. **Deep Learning Challenge — Dinosaur Image Classification**
2. **Machine Learning Challenge — Campus Recruitment Analysis and Preprocessing**

Together, the projects cover two different AI workflows: an image-classification pipeline based on convolutional neural networks and transfer learning, and a structured-data machine learning pipeline focused on exploratory analysis, data cleaning, encoding, scaling, and feature preparation.

---

## Project Links

- [Deep Learning Challenge](https://github.com/JennyDretaki/AI/tree/main/DL%20Challenge%20-%20AI)
- [Machine Learning Challenge](https://github.com/JennyDretaki/AI/tree/main/ML%20Challenge%20-%20AI)

---

## Technology Overview

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, Pillow |
| Deep Learning Architectures | Custom CNN, MobileNetV2, ResNet50, VGG16, EfficientNetB0 |
| Data Formats | CSV, JSON, JPG, JPEG, PNG |

---

# Deep Learning Challenge

## Dinosaur Image Classification

The Deep Learning project addresses a **multi-class image classification problem**. The objective is to classify dinosaur images into one of seven species using both a custom convolutional neural network and transfer-learning architectures pretrained on ImageNet.

### Classes

The classification task contains seven labels:

| Label | Dinosaur Species |
|---:|---|
| 0 | Ankylosaurus |
| 1 | Diplodocus |
| 2 | Parasaurolophus |
| 3 | Stegosaurus |
| 4 | Tyrannosaurus Rex |
| 5 | Triceratops |
| 6 | Velociraptor |

---

## Dataset

The Deep Learning dataset is stored externally because of its size.

[View / Download the Deep Learning Dataset](https://drive.google.com/drive/folders/1HSTkGodOWlalCf-e-elltCcVzfQeP3jD?usp=drive_link)

The training data is organized into species-specific directories, while the test data contains images identified by ID for final prediction and submission generation.

Expected structure:

```text
DL Challenge - AI
├── Data
│   ├── train
│   │   └── train
│   │       ├── Ankylosaurus
│   │       ├── Diplodocus
│   │       ├── Parasaurolophus
│   │       ├── Stegosaurus
│   │       ├── Tyrannosaurus Rex
│   │       ├── Triceratops
│   │       └── Velociraptor
│   └── test
│       └── test
├── EDA.ipynb
├── model.ipynb
├── modeling.ipynb
├── requirements.txt
└── submission.csv
```

---

## Exploratory Data Analysis

The EDA notebook examines both dataset quality and image characteristics before training.

The analysis includes:

- Counting images per dinosaur class
- Detecting missing class directories
- Verifying image integrity with Pillow
- Detecting invalid or corrupted image files
- Visualizing class distribution
- Displaying representative samples from every class
- Measuring image width and height
- Examining RGB color distributions
- Checking test-image availability
- Removing invalid image files when detected
- Saving metadata for reuse during modeling

The EDA stage produces or defines visual outputs such as:

```text
class_distribution.png
sample_images.png
image_dimensions.png
color_distribution.png
data_metadata.json
```

---

## Image Preprocessing

Images are resized to:

```text
224 x 224 pixels
```

with a batch size of:

```text
32
```

Pixel values are normalized using:

```python
rescale=1./255
```

A validation split of:

```text
20%
```

is used, leaving approximately 80% of the prepared training data for model training.

### Data Augmentation

The training pipeline applies augmentation to improve generalization. Techniques used across the notebooks include:

- Rotation up to 20 degrees
- Width shifting
- Height shifting
- Zoom
- Horizontal flipping
- Shear transformations in the EDA preparation pipeline
- Nearest-neighbor filling for transformed pixels

The test data is only rescaled and is not randomly augmented.

---

## Deep Learning Models

The comparative modeling notebook implements five architectures.

### 1. Custom CNN

The custom convolutional neural network is built from scratch.

Architecture:

```text
Input: 224 x 224 x 3

Conv2D 32, 3x3, ReLU
MaxPooling2D

Conv2D 64, 3x3, ReLU
MaxPooling2D

Conv2D 128, 3x3, ReLU
MaxPooling2D

Conv2D 256, 3x3, ReLU
MaxPooling2D

Conv2D 256, 3x3, ReLU
MaxPooling2D

Flatten
Dense 256, ReLU
Dropout 0.5
Dense 7, Softmax
```

### 2. MobileNetV2

MobileNetV2 is loaded with ImageNet weights and without its original classification head.

```text
MobileNetV2 ImageNet Base
Base Layers Frozen
GlobalAveragePooling2D
Dropout 0.5
Dense 7, Softmax
```

### 3. ResNet50

```text
ResNet50 ImageNet Base
Base Layers Frozen
GlobalAveragePooling2D
Dropout 0.5
Dense 7, Softmax
```

### 4. VGG16

```text
VGG16 ImageNet Base
Base Layers Frozen
GlobalAveragePooling2D
Dropout 0.5
Dense 7, Softmax
```

### 5. EfficientNetB0

```text
EfficientNetB0 ImageNet Base
Base Layers Frozen
GlobalAveragePooling2D
Dropout 0.5
Dense 7, Softmax
```

The four transfer-learning models reuse pretrained ImageNet features while keeping the convolutional base frozen during the initial training stage.

---

## Focused MobileNetV2 Pipeline

A second modeling notebook also implements a focused MobileNetV2 workflow.

Its custom classification head is:

```text
MobileNetV2
GlobalAveragePooling2D
Dense 512, ReLU
Dropout 0.5
Dense 7, Softmax
```

This provides a simpler transfer-learning experiment alongside the broader five-model comparison.

---

## Training Configuration

### Comparative Model Training

All five models are compiled with:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |
| Maximum Epochs | 15 |
| Early Stopping Monitor | Validation Loss |
| Early Stopping Patience | 3 |
| Restore Best Weights | Yes |

### Focused MobileNetV2 Training

The focused MobileNetV2 notebook uses:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Metric | Accuracy |
| Maximum Epochs | 20 |
| Early Stopping Patience | 5 |
| Restore Best Weights | Yes |

---

## Model Evaluation

The Deep Learning pipeline evaluates the trained models using:

- Validation accuracy
- Training and validation loss
- Training and validation accuracy curves
- Confusion matrix
- Precision
- Recall
- F1-score
- Classification report

The comparative notebook selects the best architecture according to the highest recorded validation accuracy.

The supplied modeling notebook defines the complete evaluation workflow, but does not contain executed numerical model-comparison outputs. For that reason, this README does not claim a specific final accuracy value.

---

## Test Predictions and Submission

After model selection, the best model is used to predict the test set.

Predicted probabilities are converted to class indices using:

```python
np.argmax(...)
```

The final submission follows the format:

```text
id,label
```

and is saved as:

```text
submission.csv
```

---

## Deep Learning Libraries

The project provides pinned dependency versions:

| Library | Version | Purpose |
|---|---:|---|
| NumPy | 1.24.3 | Numerical operations and arrays |
| Pandas | 2.0.3 | DataFrames and submission generation |
| Matplotlib | 3.7.2 | Training curves and EDA plots |
| Seaborn | 0.12.2 | Statistical plots and confusion-matrix visualization |
| OpenCV | 4.8.0.76 | Image analysis and image-dimension inspection |
| Pillow | 10.0.0 | Image validation and integrity checking |
| TensorFlow | 2.13.0 | Deep learning, Keras, transfer learning and training |
| Scikit-learn | 1.3.0 | Confusion matrix and classification metrics |

---

# Machine Learning Challenge

## Campus Recruitment Dataset

The Machine Learning project works with structured campus-recruitment data and focuses on understanding the variables, identifying patterns, and preparing the dataset for downstream machine learning modeling.

The project is organized around two main stages:

1. Exploratory Data Analysis
2. Preprocessing and Feature Engineering

---

## Dataset Structure

The analyzed training data contains 15 columns.

| Feature | Description / Role |
|---|---|
| `id` | Record identifier |
| `gender` | Categorical feature |
| `ssc_p` | Secondary-school percentage |
| `ssc_b` | Secondary-school board |
| `hsc_p` | Higher-secondary percentage |
| `hsc_b` | Higher-secondary board |
| `hsc_s` | Higher-secondary specialization/stream |
| `degree_p` | Degree percentage |
| `degree_t` | Degree type |
| `workex` | Work-experience indicator |
| `etest_p` | Employability-test percentage |
| `specialisation` | MBA specialization |
| `mba_p` | MBA percentage |
| `status` | Placement status |
| `salary` | Salary value |

The target explored in the EDA is:

```text
Placed
Not Placed
```

The provided challenge files also include:

```text
train_campusrecruit.csv
test_campusrecruit.csv
cleaned_train_campusrecruit.csv
sample_submission.csv
```

The sample submission uses:

```text
id,status
```

which reflects the placement-classification objective.

---

## Machine Learning Exploratory Data Analysis

The EDA notebook uses Pandas, Matplotlib, and Seaborn to examine the recruitment dataset.

### Analysis Performed

- Dataset preview
- Data types and general dataset information
- Descriptive statistics
- Missing-value inspection
- Placement-status distribution
- Numerical-feature distributions
- Categorical-feature distributions
- Feature-correlation analysis

### Numerical Features Explored

```text
ssc_p
hsc_p
degree_p
etest_p
mba_p
```

Salary is also included in the descriptive statistical analysis.

### Categorical Features Explored

```text
gender
ssc_b
hsc_b
hsc_s
degree_t
workex
specialisation
```

### Visualizations

The notebook creates:

- Placement status count plot
- Histograms for numerical variables
- Count plots for categorical variables
- Correlation heatmap for numeric columns

### EDA Findings

The notebook identifies three main observations:

1. The placement classes are imbalanced, with more students classified as `Placed` than `Not Placed`.
2. Academic-performance variables show meaningful correlation patterns.
3. Categorical variables show identifiable trends, including relatively limited work experience.

The analyzed complete dataset contains 700 records.

---

## Preprocessing and Feature Engineering

The second Machine Learning notebook prepares the data for future model training.

### 1. Missing Values

Rows containing missing values are removed:

```python
train.dropna(inplace=True)
```

### 2. Categorical Encoding

Categorical variables are transformed into numerical values using Scikit-learn's:

```python
LabelEncoder
```

Encoded columns:

```text
gender
ssc_b
hsc_b
hsc_s
degree_t
workex
status
specialisation
```

Each fitted encoder is stored in a dictionary for possible later reuse.

### 3. Numerical Feature Scaling

The following features are standardized using:

```python
StandardScaler
```

Columns:

```text
ssc_p
hsc_p
degree_p
etest_p
mba_p
```

This transforms the numerical variables onto a standardized scale suitable for many machine learning algorithms.

### 4. Salary Processing

The preprocessing notebook also performs salary-specific cleaning:

- Converts `salary` to floating-point values
- Removes rows with negative salary values
- Calculates the median of positive salary values
- Replaces salary values equal to `0.0` with the calculated median

### 5. Processed Dataset

The cleaned data is exported to:

```text
DataSet/cleaned_train_campusrecruit.csv
```

This dataset is intended to be used by the subsequent modeling stage.

---

## Machine Learning Libraries

| Library / Module | Purpose |
|---|---|
| Pandas | Data loading, inspection, manipulation and CSV export |
| NumPy | Numerical and array operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical plots and correlation visualization |
| Scikit-learn `LabelEncoder` | Categorical-variable encoding |
| Scikit-learn `StandardScaler` | Numerical-feature standardization |
| os | Directory and file-path management |

---

## Current Machine Learning Pipeline

```text
Campus Recruitment Dataset
          |
          v
Exploratory Data Analysis
          |
          v
Missing-Value Handling
          |
          v
Categorical Encoding
          |
          v
Numerical Scaling
          |
          v
Salary Cleaning
          |
          v
Cleaned Dataset
          |
          v
Prepared for Modeling
```

The provided ML notebooks cover the EDA, cleaning, encoding, scaling, and feature-preparation stages. They prepare the dataset for downstream placement-classification modeling.

---

# Repository Structure

A simplified representation of the repository is:

```text
AI
├── DL Challenge - AI
│   ├── Data
│   ├── EDA.ipynb
│   ├── model.ipynb
│   ├── modeling.ipynb
│   ├── requirements.ipynb
│   ├── requirements.txt
│   └── submission.csv
│
├── ML Challenge - AI
│   ├── DataSet
│   │   ├── train_campusrecruit.csv
│   │   ├── test_campusrecruit.csv
│   │   ├── cleaned_train_campusrecruit.csv
│   │   └── sample_submission.csv
│   ├── 1_EDA.ipynb
│   ├── 2_Preprocessing_FeatureEngineering.ipynb
│   └── Description.txt
│
└── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/JennyDretaki/AI.git
cd AI
```

---

## Deep Learning Environment

Move into the Deep Learning project:

```bash
cd "DL Challenge - AI"
```

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Download the Deep Learning dataset from:

[Google Drive Dataset](https://drive.google.com/drive/folders/1HSTkGodOWlalCf-e-elltCcVzfQeP3jD?usp=drive_link)

Place the files under the paths expected by the notebooks before running the EDA and modeling workflow.

---

## Machine Learning Environment

The Machine Learning notebooks require the main data-science libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Move into:

```bash
cd "ML Challenge - AI"
```

and open the notebooks with Jupyter Notebook, JupyterLab, or a compatible IDE.

---

# Recommended Execution Order

## Deep Learning

```text
1. EDA.ipynb
2. model.ipynb and/or modeling.ipynb
3. Generate submission.csv
```

## Machine Learning

```text
1. 1_EDA.ipynb
2. 2_Preprocessing_FeatureEngineering.ipynb
3. Use the cleaned dataset for downstream modeling
```

---

# Skills Demonstrated

These projects demonstrate practical experience with:

### Data Analysis

- Exploratory Data Analysis
- Data cleaning
- Descriptive statistics
- Missing-value analysis
- Correlation analysis
- Data visualization

### Machine Learning Preparation

- Categorical encoding
- Numerical scaling
- Feature preparation
- Structured dataset processing
- Scikit-learn preprocessing utilities

### Deep Learning

- Convolutional Neural Networks
- Image classification
- Transfer learning
- ImageNet pretrained models
- Data augmentation
- Softmax multi-class classification
- Early stopping
- Model comparison
- Model selection

### Computer Vision

- Image loading and verification
- Image-size analysis
- RGB color analysis
- Image normalization
- Image augmentation

### Evaluation

- Validation accuracy
- Training and validation curves
- Confusion matrices
- Precision
- Recall
- F1-score
- Classification reports

### Development

- Python
- Jupyter Notebook
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- Pillow
- CSV / JSON processing

---

# Academic Context

The Deep Learning notebooks identify the work as part of the **Artificial Intelligence course at Thomas More**.

The comparative Dinosaur Image Classification notebook lists the following team members:

- Evgenia Dretaki
- Pierina Lopez
- Gabriela Betancourth

---

# Notes

- The Deep Learning dataset is not stored directly in the repository and must be downloaded separately.
- Local file paths may need to be adjusted depending on where the repository and dataset are stored.
- Deep Learning training time depends heavily on hardware availability.
- The Deep Learning notebooks define evaluation and model-selection logic, but the supplied notebook files do not contain executed final accuracy values; therefore, no unsupported performance score is reported here.
- The Machine Learning material currently focuses on analysis and preprocessing, producing a cleaned dataset ready for a later modeling stage.
