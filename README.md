# 🩺 Skin Cancer Detection & Dashboard

## 📌 Overview
This project is a **deep learning-based skin cancer detection system** with an interactive **dashboard** for visualizing results and making predictions.  
It classifies dermatoscopic images into **Benign** and **Malignant** categories.  

You can:
- **Train your own model** using the provided Jupyter Notebook (`Skin_Cancer.ipynb`)
- **Run the dashboard** using pre-trained weights for instant predictions.

## 🎥  Dashboard Demo
![Demo](https://github.com/OmerElkayam/skin-disease-prediction/blob/main/Skin_Cancer/assets/SKIN_CANCER_GIF.gif)


---

### Live at

## 📂 Project Structure
```
.
├── Skin_Cancer.ipynb     # Jupyter Notebook for training
├── app.py                # Dashboard application
├── model_weights.h5      # Pre-trained model weights
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── data/                 # Dataset folder (not included in repo)
```

---

## 🚀 Installation & Usage

### 1️⃣ Requirements
- Python **3.10** required

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Running the Dashboard
```bash
python app.py
```
After starting, open the link shown in the terminal (example):
```
http://127.0.0.1:8050
```
Click it or paste it into your browser to use the app.

---

## 🧠 Model Training

If you want to **train the model**:
1. Open `Skin_Cancer.ipynb`
2. Update the dataset path to your dataset
3. Run all cells to train the model and save the weights

If you only want to **use the dashboard**:
- Download `app.py` and `model_weights.h5`
- Start the server as shown above

---

## 📊 Model Performance

### 📈 Training & Validation
| Metric   | Training | Validation |
|----------|----------|------------|
| Accuracy | 0.8969   | 0.8494     |
| Loss     | 0.2596   | 0.3659     |

### 🧪 Test Performance
- **Accuracy:** 0.8741  
- **ROC AUC:** 0.92  

#### 🔹 Classification Report:
| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Benign    | 0.74      | 0.86   | 0.80     | 1955    |
| Malignant | 0.94      | 0.89   | 0.91     | 5130    |

- **Overall Precision:** 0.8876  
- **Overall Recall:** 0.8790  
- **Overall F1-score:** 0.8816  

---

## 📉 Confusion Matrix
![Confusion Matrix](https://github.com/OmerElkayam/skin-disease-prediction/blob/main/Skin_Cancer/assets/Screenshot%202025-08-16%20032210.png)

---

## 📈 ROC Curve
![ROC Curve](https://github.com/OmerElkayam/skin-disease-prediction/blob/main/Skin_Cancer/assets/Screenshot%202025-08-16%20032230.png)

---

## 📈 Precision-Recall Curve
![Precision-Recall Curve](https://github.com/OmerElkayam/skin-disease-prediction/blob/main/Skin_Cancer/assets/Screenshot%202025-08-16%20032248.png)

---

## 📈 Predictions on Test Data
![Precision-Recall Curve](https://github.com/OmerElkayam/skin-disease-prediction/blob/main/Skin_Cancer/assets/Screenshot%202025-08-16%20032340.png)

---

## 📚 Dataset
Dataset: **[ISIC Skin Cancer Dataset](https://www.isic-archive.com/)**  
Place the dataset in the `data/` directory before training.

---

## 📜 License
Licensed under the **MIT License** — you are free to use and modify this project.

---

## 🙌 Acknowledgments
- [ISIC Archive](https://www.isic-archive.com/) for the dataset  
- TensorFlow / Keras documentation  
- The open-source deep learning community  
