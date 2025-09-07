from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model and scaler
# TODO: Load the trained model from 'model.pkl' using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# TODO: Load the scaler from 'scaler.pkl' using pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    prediction_proba = None
    
    if request.method == 'POST':
        # TODO: Retrieve input values from the form (geography, gender, age, balance, credit_score, estimated_salary, tenure, num_of_products, has_cr_card, is_active_member)

        geography = request.form['geography']
        gender = request.form['gender']
        age = float(request.form['age'])
        balance = float(request.form['balance'])
        credit_score = float(request.form['credit_score'])
        estimated_salary = float(request.form['estimated_salary'])
        tenure = float(request.form['tenure'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])

        geography_germany = 1 if geography == 'Germany' else 0
        geography_spain = 1 if geography == 'Spain' else 0
        gender_male = 1 if gender == 'Male' else 0
        
        # TODO: Create a pandas DataFrame with the input values matching the training feature columns
        input_data = pd.DataFrame([[
            credit_score, age, tenure, balance, num_of_products,
            has_cr_card, is_active_member, estimated_salary,
            geography_germany, geography_spain, gender_male
        ]], columns=[
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_Germany', 'Geography_Spain', 'Gender_Male'
        ])
        # TODO: Ensure the columns are ordered in the same way as the model expects
        
        # TODO: Scale the input using the loaded scaler
        scaled_input = scaler.transform(input_data)
        
        # TODO: Use the loaded model to predict and get the probability
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]

    return render_template('index.html', prediction=prediction, prediction_proba=prediction_proba)

if __name__ == '__main__':
    # Run Flask app on port 3000
    app.run(host = '0.0.0.0', port = 3000, debug = True)