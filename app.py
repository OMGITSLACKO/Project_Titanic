from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model (ensure this file exists in your project directory)
model_path = "best_tuned_mlp_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' was not found in the directory.")
model = joblib.load(model_path)

# HTML template with dark theme and golden accents
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survivor Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1e1e1e;
            color: #e0e0e0;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 700px;
            margin: 50px auto;
            background: #2e2e2e;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            border-left: 5px solid #d4af37; /* Golden accent */
            border-radius: 5px;
        }
        h1 {
            text-align: center;
            color: #d4af37;
        }
        form {
            display: flex;
            flex-direction: column;
        }
        label {
            margin-top: 15px;
            font-weight: bold;
        }
        select, input[type="number"] {
            padding: 10px;
            font-size: 16px;
            margin-top: 5px;
            background: #3e3e3e;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 4px;
        }
        select:focus, input[type="number"]:focus {
            outline: none;
            border-color: #d4af37;
        }
        input[type="submit"] {
            margin-top: 25px;
            padding: 15px;
            font-size: 18px;
            background: #d4af37;
            color: #1e1e1e;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s ease;
        }
        input[type="submit"]:hover {
            background: #c69e35;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #3e3e3e;
            border-left: 5px solid #d4af37;
            border-radius: 5px;
        }
        .message {
            font-size: 20px;
            text-align: center;
        }
        .feature-list {
            margin-top: 15px;
            list-style-type: none;
            padding: 0;
        }
        .feature-list li {
            margin-bottom: 5px;
        }
        a {
            display: block;
            text-align: center;
            margin-top: 30px;
            text-decoration: none;
            color: #d4af37;
            font-weight: bold;
        }
        a:hover {
            text-decoration: underline;
        }
        /* Responsive Design */
        @media (max-width: 600px) {
            .container {
                margin: 20px;
                padding: 20px;
            }
            input[type="submit"] {
                font-size: 16px;
                padding: 12px;
            }
            .message {
                font-size: 18px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Titanic Survivor Predictor</h1>
        <p>Please enter the following details to predict survival:</p>
        <form method="post">
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" min="0" max="100" required>
            
            <label for="sex">Sex:</label>
            <select id="sex" name="sex" required>
                <option value="" disabled selected>Select Sex</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
            </select>
            
            <label for="pclass">Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd):</label>
            <select id="pclass" name="pclass" required>
                <option value="" disabled selected>Select Class</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
            </select>
            
            <label for="sibsp">Number of Siblings/Spouses Aboard:</label>
            <select id="sibsp" name="sibsp" required>
                <option value="" disabled selected>Select Number</option>
                {% for i in range(0, 11) %}
                <option value="{{i}}">{{i}}</option>
                {% endfor %}
            </select>
            
            <label for="fare">Fare:</label>
            <input type="number" id="fare" name="fare" step="0.01" min="0" placeholder="e.g., 32.00" required>
            <p style="color: #d4af37; font-size: 14px;">*Median Fare: $32.00</p>

            
            <input type="submit" value="LET'S SEE!">
        </form>
        
        {% if prediction is defined %}
            <div class="result">
                <ul class="feature-list">
                    <li><strong>Age:</strong> {{ age }}</li>
                    <li><strong>Sex:</strong> {{ sex }}</li>
                    <li><strong>Passenger Class:</strong> {{ pclass }}</li>
                    <li><strong>Number of Siblings/Spouses Aboard:</strong> {{ sibsp }}</li>
                    <li><strong>Fare:</strong> {{ fare }}</li>
                </ul>
                <div class="message">
                    {% if prediction == 0 %}
                        <p>O-o... Looks like this is a DiCaprio situation :O</p>
                    {% else %}
                        <p>Yaaay, this person is a floater! ^_^</p>
                    {% endif %}
                </div>
            </div>
        {% endif %}
        
        <a href="/">Back to Home</a>
    </div>
</body>
</html>
"""

# Index route with basic instructions
@app.route('/')
def index():
    return render_template_string(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Titanic Survivor Predictor</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #1e1e1e;
                    color: #e0e0e0;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    max-width: 700px;
                    margin: 50px auto;
                    background: #2e2e2e;
                    padding: 30px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
                    border-left: 5px solid #d4af37; /* Golden accent */
                    border-radius: 5px;
                    text-align: center;
                }
                h1 {
                    color: #d4af37;
                }
                a {
                    display: inline-block;
                    margin-top: 30px;
                    padding: 10px 20px;
                    background: #d4af37;
                    color: #1e1e1e;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                    transition: background 0.3s ease;
                }
                a:hover {
                    background: #c69e35;
                }
                /* Responsive Design */
                @media (max-width: 600px) {
                    .container {
                        margin: 20px;
                        padding: 20px;
                    }
                    a {
                        padding: 8px 16px;
                        font-size: 16px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to the Titanic Survivor Predictor!</h1>
                <p>Use the button below to enter passenger details and predict survival.</p>
                <a href="/simulate">Start Prediction</a>
            </div>
        </body>
        </html>
        """
    )

# Simulated environment: a webpage with a dark theme and golden accent.

@app.route('/simulate', methods=['GET', 'POST'])
def simulate():
    if request.method == 'POST':
        try:
            # Retrieve form data
            age = float(request.form.get('age'))
            sex = request.form.get('sex').lower()
            pclass = int(request.form.get('pclass'))
            sibsp = int(request.form.get('sibsp'))
            fare = float(request.form.get('fare'))
            
            # Feature engineering
            sex_male = 1 if sex == 'male' else 0
            pclass_3 = 1 if pclass == 3 else 0
            pclass_2 = 1 if pclass == 2 else 0
            child = 1 if age < 21 else 0
            
            # Prepare feature array in the order expected by the model
            features = [sex_male, pclass_3, sibsp, fare, child, pclass_2]
            features_array = np.array(features).reshape(1, -1)
            
            # Debug: Print features being passed to the model
            print("Features passed to the model:", features_array)
            
            # Make prediction
            prediction = model.predict(features_array)[0]  # 0 or 1
            
            # Debug: Print prediction
            print("Model prediction:", prediction)
            
            # Render the template with prediction
            return render_template_string(
                html_template,
                prediction=prediction,
                age=age,
                sex=sex.capitalize(),
                pclass=pclass,
                sibsp=sibsp,
                fare=fare
            )
        except Exception as e:
            # In case of error, render the form with an error message
            return render_template_string(
                html_template,
                prediction_text=f"Error: {str(e)}"
            )
    else:
        # GET request: render the empty form
        return render_template_string(html_template)

# REST API endpoint: expects JSON input.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required_features = ['Age', 'Sex', 'Pclass', 'SibSp', 'Fare']
        if not all(feature in data for feature in required_features):
            return jsonify({"error": f"Missing one of the required features: {required_features}"}), 400
        
        # Retrieve and process features
        age = float(data['Age'])
        sex = data['Sex'].lower()
        pclass = int(data['Pclass'])
        sibsp = int(data['SibSp'])
        fare = float(data['Fare'])
        
        # Feature engineering
        sex_male = 1 if sex == 'male' else 0
        pclass_3 = 1 if pclass == 3 else 0
        pclass_2 = 1 if pclass == 2 else 0
        child = 1 if age < 21 else 0
        
        # Prepare feature array in the order expected by the model
        features = [sex_male, pclass_3, sibsp, fare, child, pclass_2]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]  # 0 or 1
        
        # Return prediction
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
