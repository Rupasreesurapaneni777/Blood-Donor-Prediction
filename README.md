# 🩸 Blood Donor Prediction

A Machine Learning project that predicts whether a person is likely to **donate blood again** based on their previous blood donation history.

## 📌 Project Overview

The **Blood Donor Prediction** project uses historical blood donation data to identify potential repeat blood donors. The project includes data preprocessing, exploratory data analysis, feature selection, model training, prediction, and performance evaluation.

A **Logistic Regression** classification algorithm is used to predict the likelihood of future blood donation.

## 🎯 Objectives

* Predict whether a donor is likely to donate blood again.
* Analyze historical blood donation patterns.
* Apply data preprocessing and Machine Learning techniques.
* Build a simple web application for making predictions.

## 🚀 Key Features

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature selection
* Training and testing of the Machine Learning model
* Blood donation prediction
* Model performance evaluation
* Flask-based web application
* Simple HTML/CSS user interface

## 🛠️ Technologies Used

| Technology          | Purpose                             |
| ------------------- | ----------------------------------- |
| Python              | Programming and Machine Learning    |
| Pandas              | Data manipulation and preprocessing |
| NumPy               | Numerical operations                |
| Scikit-learn        | Machine Learning                    |
| Logistic Regression | Classification model                |
| Jupyter Notebook    | Data analysis and experimentation   |
| Flask               | Web application                     |
| HTML & CSS          | Frontend interface                  |

## 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary classification.

The model takes historical blood donation information as input and predicts whether the donor is likely to donate blood again.

### Model Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Performance

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning & Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Logistic Regression Model
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Flask Web Application
```

## 📂 Project Structure

```text
blood-donor-prediction/
│
├── dataset/
│   └── blood_donation.csv
│
├── templates/
│   └── index.html
│
├── app.py
├── model.py
├── blood_donor_prediction.ipynb
├── requirements.txt
└── README.md
```

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Rupasreesurapaneni777/blood-donor-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd blood-donor-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
python app.py
```

### 5. Open the Application

Open the local URL displayed in the terminal, usually:

```text
http://127.0.0.1:5000/
```

## 🖥️ Project Output

The Flask application provides a simple interface where users can enter donor-related information and receive a prediction.

![Blood Donor Prediction Output](https://github.com/user-attachments/assets/893ad757-9c65-40ab-b4d8-432edce3bbaf)

## 💡 Applications

This project demonstrates how Machine Learning can be used to:

* Identify potential repeat blood donors.
* Analyze donor behavior.
* Support blood donation management.
* Assist organizations in targeting potential donors.

## 🔮 Future Improvements

* Compare Logistic Regression with other classification algorithms.
* Improve model performance through hyperparameter tuning.
* Add more donor-related features.
* Improve the user interface.
* Add data visualization to the web application.
* Deploy the application using a cloud platform.
* Add real-time prediction capabilities.

## 👩‍💻 Skills Demonstrated

**Python • Machine Learning • Logistic Regression • Pandas • NumPy • Scikit-learn • Data Preprocessing • EDA • Flask • HTML • CSS**

---

⭐ If you find this project useful, feel free to **star the repository**!
