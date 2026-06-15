from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    age = int(request.form['age'])

    # Prepare data for prediction
    features = np.array([[area, bedrooms, age]])

    # Predict
    prediction = model.predict(features)

    # Convert to integer
    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Estimated House Price: ₹ {output}'
    )

# Run application
if __name__ == "__main__":
    app.run(debug=True)