Stock Buy Recommendation System

A machine learning–powered web application built with Streamlit that predicts whether a stock is worth buying based on its fundamental indicators.

🔥 Features

📊 User-friendly Streamlit interface with modern UI.

🤖 Multiple ML models trained (Logistic Regression, Random Forest, KNN, Gradient Boosting) with cross-validation.

🏆 Automatically selects the best-performing model.

💾 Model persistence using joblib for reliable storage/loading.

✅ Real-time stock recommendation: Buy ✅ or Not Buy ❌.

📂 Project Structure
├── data/
│   └── top1kstocks.csv       # Dataset with stock fundamentals
├── model.pkl                 # Saved best ML model
├── streamlit_app.py          # Streamlit UI code
├── train_model.py            # Training + model selection script
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation

⚙️ Installation

Clone this repository:

git clone https://github.com/yourusername/stock-recommendation.git
cd stock-recommendation


Create a virtual environment & activate it:

python -m venv venv
venv\Scripts\activate   # on Windows
source venv/bin/activate # on Mac/Linux


Install dependencies:

pip install -r requirements.txt

🏋️‍♂️ Train the Model

Run the training script to train multiple models, evaluate accuracy, and save the best model:

python train_model.py


This generates a model.pkl file using joblib.

🚀 Run the Streamlit App

Start the web app:

streamlit run streamlit_app.py


This will open the app in your browser at http://localhost:8501.

🖼️ Example Usage

Enter stock fundamentals like CMP, P/E, Market Cap, ROCE, Annual Profit, etc.

Click Predict.

The app will recommend:

✅ Buy if the model predicts good growth.

❌ Not Buy otherwise.

📦 Requirements

Python 3.8+

scikit-learn

pandas

numpy

streamlit

joblib

matplotlib

Install them with:

pip install -r requirements.txt

🌟 Future Improvements

Add deep learning models with TensorFlow/PyTorch.

Integrate real-time stock market API for live predictions.

Advanced visualization of feature importance.

Deploy to Streamlit Cloud / Heroku / AWS.