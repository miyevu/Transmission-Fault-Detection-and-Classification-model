# Transmission Fault Detection and Classification Model

## Project Overview

This project aims to develop a robust machine learning model for detecting and classifying faults in transmission systems. Using the Artificial Neural Network(ANN) algorithms, the model is designed to accurately identify different types of shunt faults, enhancing the reliability and safety of power transmission networks.

## Features

- **Fault Detection**: Identify the presence of faults in the transmission system.
- **Fault Classification**: Classify faults into specific categories (e.g., Line-to-Ground, Line-to-Line with Ground).
- **High Accuracy**: Achieved over 98% accuracy in fault detection and 86% in fault classification.
- **Model Training**: Utilizes a deep learning approach with an emphasis on feature extraction and data preprocessing.

## Project Structure

- **`ANN_project/dataset/`**: Contains datasets used for training and testing.
- **`ANN_project/notebooks/`**: Holds Jupyter notebooks for data exploration and model training.
- **`README.md`**: This file.
- **`requirements.txt`**: List of Python dependencies.

## Installation

Clone the repository and install the required packages:

```bash
git clone git@github.com:miyevu/Transmission-Fault-Detection-and-Classification-model.git
cd Transmission-Fault-Detection-and-Classification-model
pip install -r requirements.txt
```

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Data

The dataset used for this project contains various features related to transmission line conditions, including current and voltage measurements, and fault indicators. The data is split into training (80%) and validation (20%) sets.
<!-- 
## Model Training

To train the model, run the following script:

```bash
python scripts/train_model.py
```

This script handles data loading, preprocessing, model training, and evaluation. The model's performance metrics are logged and saved for further analysis.

## Evaluation

Evaluate the model's performance with the following script:

```bash
python scripts/evaluate_model.py
```

This script will output the accuracy, precision, recall, and F1-score of the model on the validation set. -->

<!-- ## Usage

To make predictions on new data, use the `predict.py` script:

```bash
python scripts/predict.py --input_path path/to/new/data.csv --model_path models/best_model.h5
``` -->

## Results

- **Fault Detection Accuracy**: 98%
- **Fault Classification Accuracy**: 86%
<!-- - **Model Architecture**: [Details on the model architecture, e.g., number of layers, activation functions, etc.] -->

## Contributing

Contributions are welcome! Please feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

<!-- ## License

Distributed under the MIT License. See `LICENSE` for more information. -->

## Contact

<!-- - **Author**: [Your Name](mailto:your.email@example.com) -->
- **GitHub**: [miyevu](https://github.com/miyevu)
