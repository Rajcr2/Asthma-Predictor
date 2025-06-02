# Asthma Risk Predictor

## Introduction

I have developed an asthma risk predictor system for both general users and clinical patients/professionals. It assesses the likelihood of asthma based on basic symptoms and advanced clinical metrics given by user and provides results along with Probability of the Risk. 

### Key Features

1. **Dual User Interface**
2. **Model Driven Risk Prediction**
   - **XGBoost** Model for general **Symptom based assessment**.
   - **CatBoost** Model for clinical evaluation using metrics like **FEV1/FVC Ratio**.

### Objectives

The primary goal of this project is to develope a asthma risk predictor system that can :

1. Predict near to perfect Asthma Risk along with Probability. 
2. Incorporate general health indicators, symptoms and clinical metrics based on that provide accurate results.
3. Adapt to both general and clinical users or healthcare professionals which provides Advice to general users along with results while clear information to clinical professionals.

### Technologies Used

- **Machine Learning** : XGBoost, CatBoost
- **Frontend** : Streamlit
- **Testing** : Pytest
- **Language** : Python 3.10+


### Prerequisites
To run this project, you need to install the following libraries:
### Required Libraries

- **Python 3.10+**
- **Pandas**: This library performs data manipulation and analysis also provides powerful data structures like dataframes.
- **XGBoost**: An optimized gradient boosting library designed for high performance and speed in structured/tabular data tasks.
- **CatBoost**: A gradient boosting algorithm developed by Yandex, built to handle categorical features automatically.
- **Pytest**: Python testing framework used for writing simple to complex test cases for applications and libraries.
- **Streamlit**: Streamlit is a framework that builds interactive, data-driven web applications directly in python.  

Other Utility Libraries : **numpy**.

### Installation

   ```
   pip install pandas
   pip install scikit-learn
   pip install numpy
   pip install streamlit
   pip install xgboost
   pip install catboost
   pip install pytest
   ```

### Procedure

1.   Create new directory **'Asthma Predictor'**.
2.   Inside that directory/folder create new environment.
   
   ```
   python -m venv asthp
   ```

  Now, activate this **'asthp'** venv.
  
4.   Clone this Repository :

   ```
   git clone https://github.com/Rajcr2/Asthma-Predictor.git
   ```
5.   Now, Install all mentioned required libraries in your environment.
6.   After, that Run **'main.py'** file from Terminal. To activate the dashboard on your browser.
   ```
   streamlit run src/main.py
   ``` 
7.   Now, move to your browser.
8.   First select user type whether you are General or Clinical if you have performed lab test already then choose Clinical user type otherwise stick to General user type.
9.   Enter basic health details as an input from user and click on **'Predict Asthma Risk'** button if you are clinical user then click on **'Clinical Asthma Risk Prediction'**.
10.  After that wait for results within seconds you will see whether you are **at risk of asthma or not** along with **'Probability'**.

### Output

#### General Assesment :

https://github.com/user-attachments/assets/8e57d47e-599e-406b-815f-bf19e53e042b


#### Clinical Assesment :

https://github.com/user-attachments/assets/75a9b34b-83da-4bff-8ed9-e360e27a2387


### Conclusion

This project demonstrates how machine learning models like XGBoost and CatBoost can be effectively used to assess the asthma risk in both general and clinical scenarios, supporting early diagnosis and public health awareness.








