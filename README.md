🩺 Skin Cancer Detection & Dashboard
📌 Overview

This project provides a deep learning-based skin cancer detection system along with a dashboard for visualization and predictions.
It can classify Benign and Malignant skin lesions using dermatoscopic images.

You can:

Retrain the model by running the Jupyter Notebook (Skin_Cancer.ipynb)

Run the dashboard for predictions using the trained model weights

📂 Project Structure
.
├── Skin_Cancer.ipynb   # Jupyter Notebook for model training
├── app.py              # Dashboard application
├── model_weights.h5    # Pre-trained model weights
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── data/               # Dataset folder (not included due to size)

🚀 Installation & Usage
1️⃣ Requirements

Python 3.10 required

2️⃣ Install Dependencies

Run the following command:

pip install -r requirements.txt

3️⃣ Running the Dashboard

To start the server:

python app.py


Once the server starts, you will see a link in the terminal:

http://127.0.0.1:8050


Click the link or paste it into your browser to use the app.

🧠 Model Training

If you want to train your own model:

Open Skin_Cancer.ipynb

Change the dataset path in the notebook to your own dataset

Run all cells to train and save the model weights

If you just want to use the dashboard for predictions:

Download app.py and the model_weights.h5 file

Start the server as explained above

📊 Model Performance
📈 Training Results

Train Accuracy: 0.8969

Train Loss: 0.2596

Validation Accuracy: 0.8494

Validation Loss: 0.3659

🧪 Test Results

Test Accuracy: 0.8741

ROC Curve AUC: 0.92

🔹 Classification Report:
Class	Precision	Recall	F1-score	Support
Benign	0.74	0.86	0.80	1955
Malignant	0.94	0.89	0.91	5130

Overall Precision: 0.8876

Overall Recall: 0.8790

Overall F1-score: 0.8816

📉 Confusion Matrix


(Replace with actual image if available)

📚 Dataset

Dataset used: ISIC Skin Cancer Dataset
Download the dataset and place it in the data/ directory before training.

📜 License

This project is licensed under the MIT License — feel free to use and modify.

🙌 Acknowledgments

ISIC Archive for the dataset

TensorFlow / Keras documentation

Open-source deep learning community
