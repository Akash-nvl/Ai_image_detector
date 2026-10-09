# 🔍 AI Image Forensics Studio

LIVELINK:https://aiimagedetector-xrtwugcxpabt4uhr4kycgb.streamlit.app/

### Intelligent Image Authenticity Analyzer

AI Image Forensics Studio is a web application built using **Python, Streamlit, TensorFlow, and MobileNetV2**. It analyzes uploaded images and predicts whether they resemble the **AI-generated (FAKE)** or **REAL** classes learned by a trained deep learning model.

The application provides an interactive dashboard, image preview, prediction confidence, image details, and analysis history.

## 🚀 Features

* **AI Image Classification:** Classifies images into AI/FAKE or REAL categories.
* **Image Upload:** Supports JPG, JPEG, and PNG images.
* **Image Preview:** Displays the uploaded image before analysis.
* **Deep Learning Model:** Uses a trained MobileNetV2 model.
* **Confidence Score:** Displays the predicted class and confidence percentage.
* **Prediction Details:** Shows class probabilities, image dimensions, and inference information.
* **Analysis History:** Displays previous predictions during the current session.
* **Model Information:** Provides information about the architecture and dataset.
* **Interactive Dashboard:** Built with Streamlit and custom CSS styling.

## 🛠️ Technologies Used

| Technology         | Purpose                                 |
| ------------------ | --------------------------------------- |
| Python             | Main programming language               |
| Streamlit          | Web application interface               |
| TensorFlow / Keras | Deep learning and model inference       |
| MobileNetV2        | Image classification architecture       |
| NumPy              | Image array processing                  |
| Pillow             | Image loading, conversion, and resizing |
| Pandas             | Analysis history table                  |
| CIFAKE             | Training and evaluation dataset         |

## 🧠 Model Overview

The application uses MobileNetV2 for binary image classification.

* **Input size:** 224 × 224 pixels
* **Input format:** RGB images
* **Preprocessing:** MobileNetV2 `preprocess_input`
* **Output activation:** Sigmoid
* **Classes:** AI/FAKE and REAL
* **Saved model:** `cifake_mobilenetv2.keras`

The application interprets the sigmoid output as the probability of the REAL class. Predictions depend on the model's learned class mapping.

**Note:** The model's prediction is an estimate, not definitive proof of whether an image is authentic or AI-generated. Performance may vary on images unlike those in the training dataset.

## ⚙️ Installation and Setup

### 1. Install Python

Install a compatible Python version. This project was developed with Python 3.10.

### 2. Open the project folder

Open the project folder in Visual Studio Code.

### 3. Create a virtual environment

Open the VS Code terminal and run:

```powershell
python -m venv venv
```

### 4. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

Make sure `requirements.txt` contains the required packages, including:

```text
tensorflow
numpy
matplotlib
scikit-learn
Pillow
streamlit
pandas
```

Install them using:

```powershell
python -m pip install -r requirements.txt
```

### 6. Add the trained model

Place `cifake_mobilenetv2.keras` in the same folder as `app.py` and `model_utils.py`.

If the model file does not exist, run the training notebook first and save the trained model.

## ▶️ Run the Application

In the VS Code terminal, make sure the virtual environment is active and the terminal is in the project directory.

Run:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## 📸 How to Use

1. Open the application in your browser.
2. Select **Analyze Image** from the sidebar.
3. Upload a JPG, JPEG, or PNG image.
4. Preview the image and check its details.
5. Click **Analyze Image**.
6. Review the predicted class and confidence score.
7. Open **Prediction Details** for additional information.
8. Visit **Analysis History** to review predictions made during the current session.

## 📊 Dataset

The project uses the CIFAKE dataset, which contains images labeled FAKE and REAL.

The training notebook can be used to train and evaluate the model. A separate real-world dataset may be used for additional evaluation, provided its test images are kept separate from training.

## ⚠️ Limitations

* The classifier only predicts between the two classes learned during training.
* It does not identify the specific AI generator used.
* It does not locate image manipulations or verify authenticity using metadata.
* Prediction confidence is not a guarantee of correctness.
* Analysis history is stored in the current Streamlit session and is not permanently saved.
* Results may be less reliable for images from unfamiliar sources or domains.

## 🔮 Future Enhancements

* Add precision, recall, F1-score, and ROC-AUC evaluation.
* Integrate Grad-CAM visual explanations.
* Add batch image analysis.
* Include optional EXIF metadata inspection.
* Evaluate robustness against image compression and resizing.
* Add persistent prediction history and user accounts.

## 👨‍💻 Project Purpose

This project demonstrates how deep learning and web technologies can be combined to create an interactive image classification system for educational and research purposes.

**Disclaimer:** This tool provides model-based predictions and should not be used as the sole basis for determining image authenticity.
