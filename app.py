import os
from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        int(request.form['school']),
        int(request.form['exper']),
        int(request.form['union']),
        int(request.form['ethn']),
        int(request.form['maried']),
        int(request.form['health']),
        int(request.form['industry']),
        int(request.form['occupation']),
        int(request.form['residence'])
    ]
    
    # Convert features to array and make prediction
    prediction = model.predict([features])[0]
    
    # Return the result
    return render_template('index.html', prediction_text=f'Predicted Wage: ${prediction:.2f} per hour')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
