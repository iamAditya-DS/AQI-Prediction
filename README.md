# AQI Prediction and Analysis  

## Overview  
This is a **web-based application** for predicting and analyzing the **Air Quality Index (AQI)**.  
Built with **Streamlit** and trained ML models, the app allows you to:  
- Predict AQI values and categories based on **Country** and **City**  
- Upload custom **CSV datasets** for AQI predictions  
- Visualize air pollution levels with interactive charts  

---

## Features  
- **AQI Prediction** → Predict AQI values and categories using pre-trained models (`.pkl`)  
- **CSV Upload & Processing** → Upload your dataset and get predictions  
- **Visualization** → Pollution insights via charts and plots  
- **Streamlit UI** → Simple and user-friendly web app  

---

## Tech Stack  
- **Python**  
- **Streamlit** → Web interface  
- **Pandas** → Data processing  
- **Seaborn, Matplotlib, Plotly** → Data visualization  
- **Joblib** → Loading pre-trained models  
- **Scikit-learn** → Model training & pipelines  

---

## Installation & Usage  

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the Application
```bash
streamlit run app.py
```
The app will launch automatically in your browser

## Models & Training
Models Include:
- **classifier_pipeline.pkl → Predicts AQI Category**  
- **regressor_pipeline.pkl → Predicts AQI Value**

**Training & Analysis Notebooks:**
- **ML_Model.ipynb → Model training**
- **visualization 1-4.ipynb → Pollution data insights**

## Dataset  

The repository includes a sample dataset:  

- **AQI and Lat Long of Countries.csv** → Contains AQI values and location details for different cities and countries.  

This dataset was used to:  
- Train the machine learning models (`classifier_pipeline.pkl`, `regressor_pipeline.pkl`)  
- Provide default values for predictions when no custom dataset is uploaded  

Due to GitHub’s file size limits, the dataset is **not included in this repository**.  
If needed, you can use any AQI dataset with the following columns: 

### Columns in the dataset:  
- `Country`  
- `City`  
- `CO AQI Value`  
- `Ozone AQI Value`  
- `NO2 AQI Value`  
- `PM2.5 AQI Value`  
- `lat` (latitude)  
- `lng` (longitude)  

