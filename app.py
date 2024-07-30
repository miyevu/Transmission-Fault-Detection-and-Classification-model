from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

app = Flask(__name__)

# Load the models
model1 = load_model('ANN_project/fault_classification_model.h5')
model2 = load_model('ANN_project/fault_detection_model.h5')

# Load the training data
training_data1 = pd.read_csv('ANN_project/dataset/detect_dataset.csv')
training_data2 = pd.read_csv('ANN_project/dataset/classData.csv')

# Feature columns
features1 = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
features2 = ['G', 'C', 'B', 'A', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']

# Fit the scalers with the correct data
scaler1 = MinMaxScaler()
scaler1.fit(training_data1[features1])

scaler2 = MinMaxScaler()
scaler2.fit(training_data2[features2])

# Mapping fault type codes to fault type names
fault_types = {
    '0000': 'No Fault',
    '1000': 'Single Line to Ground A',
    '0100': 'Single Line to Ground B',
    '0010': 'Single Line to Ground C',
    '0011': 'Line-to-Line BC',
    '0101': 'Line-to-Line AC',
    '1001': 'Line-to-Line AB',
    '1010': 'Line-to-Line with Ground AB',
    '0110': 'Line-to-Line with Ground BC',
    '0111': 'Three-Phase',
    '1111': 'Three-Phase with Ground',
    '1011': 'Line A Line B to Ground Fault'
}

# Creating the Fault_Type column
training_data2['Fault_Type'] = training_data2[['G', 'C', 'B', 'A']].astype(str).agg(''.join, axis=1).map(fault_types)

# Initialize the LabelEncoder and fit it with the Fault_Type
label_encoder = LabelEncoder()
label_encoder.fit(training_data2['Fault_Type'])

def preprocess_input(data, scaler, expected_features):
    data_df = pd.DataFrame(data, columns=expected_features)
    for feature in expected_features:
        if feature not in data_df.columns:
            data_df[feature] = 0
    return scaler.transform(data_df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    input_data = np.array([[float(input_data[col]) for col in ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]])

    try:
        # Preprocess input data for the first model
        preprocessed_data1 = preprocess_input(input_data, scaler1, features1)

        # Debugging: print preprocessed data for model 1
        print(f"Preprocessed data for model 1: {preprocessed_data1}")

        # Predict with the first model
        prediction1 = model1.predict(preprocessed_data1)

        # Debugging: print raw predictions for model 1
        print(f"Raw prediction from model 1: {prediction1}")

        fault_detected = (prediction1 > 0.5).astype(int)[0][0]

        if fault_detected == 1:
            # For the second model, include default values for 'G', 'C', 'B', 'A'
            input_data_with_defaults = np.concatenate([np.zeros((1, 4)), input_data], axis=1)
            preprocessed_data2 = preprocess_input(input_data_with_defaults, scaler2, features2)

            # Debugging: print preprocessed data for model 2
            print(f"Preprocessed data for model 2: {preprocessed_data2}")

            # Predict with the second model
            prediction2 = model2.predict(preprocessed_data2)

            # Debugging: print raw predictions for model 2
            print(f"Raw prediction from model 2: {prediction2}")

            predicted_class2 = label_encoder.inverse_transform(np.argmax(prediction2, axis=1))
            fault_type = predicted_class2[0]
        else:
            fault_type = "No Fault"

        # Debugging output
        print(f"Input Data: {input_data}")
        print(f"Preprocessed Data1: {preprocessed_data1}")
        print(f"Prediction1: {prediction1}")
        # print(f"Raw prediction from model 2: {prediction2}")
        print(f"Fault Detected: {fault_detected}")
        print(f"Fault Type: {fault_type}")

        return render_template('index.html', fault_detected=fault_detected, fault_type=fault_type)

    except ValueError as e:
        return render_template('index.html', fault_detected='Error', fault_type=str(e))

if __name__ == '__main__':
    app.run(debug=True)
