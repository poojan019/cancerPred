# Cancer Prediction Model

This repository contains a machine learning model designed to predict diagnosis based on patient data.

## Project Structure

  app/: Contains the web application for user interaction
  assets/: Includes static files such as images and stylesheets
  data/: Stores datasets used for training and testing the model
  model/: Contains the trained machine learning model and realated scripts
  requirements.txt: Lists the Python dependencies required to run the project

## Prerequisites

  Ensure you have Python installed. It's recommended to use a virtual environment to manage dependencies.

## Installation
### 1.Clone the repository:
```
git clone https://github.com/poojan019/cancerPred.git
cd cancerPred
```

### 2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage
### 1. Prepare the data:
  Place your dataset in the `data/` directory. Ensure it matches the expected format used during model training.

### 2. Run the application:
```
streamlit run app/main.py
```
  This will start the web application, allowing you to input patient data and receive cancer predictions.

## Model Training
If you wish to train the model with new data:
### 1. Prepare your dataset:
  Ensure it's in the `data/` directory and formatted correctly

### 2. Train the model:
```
python model/main.py
```
  This will train the model and save the trained model file in the `model/` directory.

## Contributing
  Contributions are welcome! Please fork the repository and create a pull request with your changes.

## Access the website
  Website is currently live and accessible via below link
  https://cancerpred-lq8nue24zhmcfnbhdjefau.streamlit.app/

----------------------------------------------------------------------------------------------------------------
NOTE: This project is for educational purposes and should not be used for actual medical diagnosis
