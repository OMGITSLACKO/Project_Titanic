from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

app = Flask(__name__)

# Load the pre-trained model (ensure this file exists in your project directory)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survival_predictor_v1.joblib")
if not os.path.exists(model_path):
    logger.critical(f"The model file '{model_path}' was not found.")
    raise FileNotFoundError(f"The model file '{model_path}' was not found in the directory.")
try:
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    logger.debug(f"Pipeline Steps: {model.named_steps}")
    
    # Verify if scaler is fitted
    scaler = model.named_steps['scaler']
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        logger.info("Scaler is fitted.")
    else:
        logger.critical("Scaler is not fitted. Check the model saving process.")
        raise AttributeError("Scaler is not fitted.")
except Exception as e:
    logger.critical(f"Failed to load the model: {e}")
    raise e

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
            
            
            <label for="sex">Sex:</label>
            <select id="sex" name="sex" required>
                <option value="" disabled {% if not sex %}selected{% endif %}>Select Sex</option>
                <option value="male" {% if sex == 'male' %}selected{% endif %}>Male</option>
                <option value="female" {% if sex == 'female' %}selected{% endif %}>Female</option>
            </select>
            
            <label for="pclass">Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd):</label>
            <select id="pclass" name="pclass" required>
                <option value="" disabled {% if not pclass %}selected{% endif %}>Select Class</option>
                <option value="1" {% if pclass == '1' or pclass == 1 %}selected{% endif %}>1</option>
                <option value="2" {% if pclass == '2' or pclass == 2 %}selected{% endif %}>2</option>
                <option value="3" {% if pclass == '3' or pclass == 3 %}selected{% endif %}>3</option>
            </select>
            
            <label for="fare">Fare:</label>
            <input type="number" id="fare" name="fare" step="0.01" min="0" placeholder="e.g., 32.00" value="{{ fare if fare is not none else '' }}" required>
            <p style="color: #d4af37; font-size: 14px;">*Median Fare: $32.00</p>
            
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" min="0" max="100" value="{{ age if age is not none else '' }}" required>
            
            <label for="embarked">Embarked:</label>
            <select id="embarked" name="embarked" required>
                <option value="" disabled {% if not embarked %}selected{% endif %}>Select Port</option>
                <option value="C" {% if embarked == 'C' %}selected{% endif %}>Cherbourg</option>
                <option value="Q" {% if embarked == 'Q' %}selected{% endif %}>Queenstown</option>
                <option value="S" {% if embarked == 'S' %}selected{% endif %}>Southampton</option>
            </select>
            
            <label for="sibsp">Number of Siblings/Spouses Aboard:</label>
            <select id="sibsp" name="sibsp" required>
                <option value="" disabled {% if sibsp is none %}selected{% endif %}>Select Number</option>
                {% for i in range(0, 11) %}
                <option value="{{i}}" {% if sibsp == i or sibsp|string == i|string %}selected{% endif %}>{{i}}</option>
                {% endfor %}
            </select>
            
            <input type="submit" value="LET'S SEE!">
        </form>
        
        {% if prediction is defined %}
            <div class="result">
                <ul class="feature-list">
                    <li><strong>Age:</strong> {{ age }}</li>
                    <li><strong>Sex:</strong> {{ sex }}</li>
                    <li><strong>Passenger Class:</strong> {{ pclass }}</li>
                    <li><strong>Number of Siblings/Spouses Aboard:</strong> {{ sibsp }}</li>
                    <li><strong>Fare:</strong> £{{ fare }}</li>
                </ul>
                <div class="message">
                    {% if prediction == 0 %}
                        <p>O-o... Looks like this is a DiCaprio situation :O</p>
                    {% else %}
                        <p>Yaaay, this person is a floater! ^_^</p>
                    {% endif %}
                </div>
            </div>
        {% elif prediction_text is defined %}
            <div class="result">
                <div class="message">
                    <p>{{ prediction_text }}</p>
                </div>
            </div>
        {% endif %}
        
        {% if error %}
            <div class="result">
                <div class="message" style="color: red;">
                    <p>{{ error }}</p>
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
            age = request.form.get('age', type=float)
            sex = request.form.get('sex', type=str)
            pclass = request.form.get('pclass', type=int)
            sibsp = request.form.get('sibsp', type=int)
            fare = request.form.get('fare', type=float)
            embarked = request.form.get('embarked', type=str)
            
            # Input validation
            if fare is None or age is None or sex is None or pclass is None or sibsp is None or embarked is None:
                raise ValueError("All fields are required.")
            if fare < 0 or age < 0:
                raise ValueError("Age and Fare must be non-negative.")
            MAX_FARE = 512  # Replace with the actual max fare from training data
            fare = min(fare, MAX_FARE)  # Cap the fare at MAX_FARE
            
            # Log received inputs
            logger.debug(f"Received inputs - Age: {age}, Sex: {sex}, Pclass: {pclass}, SibSp: {sibsp}, Fare: {fare}, Embarked: {embarked}")
            
            # Feature engineering
            sex_male = 1 if sex.lower() == 'male' else 0
            pclass_3 = 1 if pclass == 3 else 0
            pclass_2 = 1 if pclass == 2 else 0
            child = 1 if age < 21 else 0
            embarked_S = 1 if embarked.upper() == 'S' else 0
            sibsp_5 = 1 if sibsp == 5 else 0
            
            # Log engineered features
            logger.debug(f"Engineered features - sex_male: {sex_male}, pclass_3: {pclass_3}, pclass_2: {pclass_2}, child: {child}, embarked_S: {embarked_S}, sibsp_5: {sibsp_5}")
            
            # Prepare feature array in the order expected by the model
            features = [sex_male, fare, pclass_3, child, pclass_2, age, embarked_S, sibsp_5]
            features_array = np.array(features).reshape(1, -1)
            logger.debug(f"Features passed to the model: {features_array}")
            
            # Make prediction using the pipeline
            prediction = model.predict(features_array)[0]  # 0 or 1
            prediction_proba = model.predict_proba(features_array)[0][1]  # Probability of survival
            logger.debug(f"Model prediction: {prediction}, Probability of Survival: {prediction_proba:.2f}")
            
            # Interpret prediction
            result = "Survived (Expected)" if prediction == 1 else "Did Not Survive (Unexpected)"
            logger.info(f"Prediction result: {result}")
            
            return render_template_string(
                html_template,
                prediction=prediction,
                age=age,
                sex=sex.capitalize(),
                pclass=pclass,
                sibsp=sibsp,
                fare=fare,
                embarked=embarked  # Pass 'embarked' here
            )
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            # Retrieve previously entered data to retain them
            age = request.form.get('age', '')
            sex = request.form.get('sex', '')
            pclass = request.form.get('pclass', '')
            sibsp = request.form.get('sibsp', '')
            fare = request.form.get('fare', '')
            embarked = request.form.get('embarked', '')
            return render_template_string(
                html_template,
                error=str(ve),
                age=age,
                sex=sex,
                pclass=pclass,
                sibsp=sibsp,
                fare=fare,
                embarked=embarked
            )
        except Exception as e:
            # Log the error with stack trace
            logger.error("Error during prediction:", exc_info=True)
            # Retrieve previously entered data to retain them
            age = request.form.get('age', '')
            sex = request.form.get('sex', '')
            pclass = request.form.get('pclass', '')
            sibsp = request.form.get('sibsp', '')
            fare = request.form.get('fare', '')
            embarked = request.form.get('embarked', '')
            return render_template_string(
                html_template,
                error="An error occurred during prediction. Please check your inputs.",
                age=age,
                sex=sex,
                pclass=pclass,
                sibsp=sibsp,
                fare=fare,
                embarked=embarked
            )
    else:
        # GET request: render the empty form with default values
        return render_template_string(
            html_template,
            prediction=None,
            age=None,
            sex=None,
            pclass=None,
            sibsp=None,
            fare=None,
            embarked=None
        )

# REST API endpoint: expects JSON input.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required_features = ['Age', 'Sex', 'Pclass', 'SibSp', 'Fare', 'Embarked']
        if not all(feature in data for feature in required_features):
            error_msg = f"Missing one of the required features: {required_features}"
            logger.warning(error_msg)
            return jsonify({"error": error_msg}), 400
        
        # Retrieve and process features
        age = float(data['Age'])
        sex = data['Sex'].lower()
        pclass = int(data['Pclass'])
        sibsp = int(data['SibSp'])
        fare = float(data['Fare'])
        embarked = data['Embarked'].upper()
        
        # Input validation
        if fare < 0 or age < 0:
            raise ValueError("Age and Fare must be non-negative.")
        MAX_FARE = 512  # Replace with the actual max fare from training data
        fare = min(fare, MAX_FARE)  # Cap the fare at MAX_FARE
        
        # Log received data
        logger.debug(f"Received JSON data: {data}")
        
        # Feature engineering
        sex_male = 1 if sex == 'male' else 0
        pclass_3 = 1 if pclass == 3 else 0
        pclass_2 = 1 if pclass == 2 else 0
        child = 1 if age < 21 else 0
        embarked_S = 1 if embarked == 'S' else 0
        sibsp_5 = 1 if sibsp == 5 else 0
        
        # Log engineered features
        logger.debug(f"Engineered features - sex_male: {sex_male}, pclass_3: {pclass_3}, pclass_2: {pclass_2}, child: {child}, embarked_S: {embarked_S}, sibsp_5: {sibsp_5}")
        
        # Prepare feature array in the order expected by the model
        features = [sex_male, fare, pclass_3, child, pclass_2, age, embarked_S, sibsp_5]
        features_array = np.array(features).reshape(1, -1)
        logger.debug(f"Features passed to the model: {features_array}")
        
        # Make prediction using the pipeline
        prediction = model.predict(features_array)[0]  # 0 or 1
        prediction_proba = model.predict_proba(features_array)[0][1]  # Probability of survival
        logger.debug(f"Model prediction: {prediction}, Probability of Survival: {prediction_proba:.2f}")
        
        # Return prediction
        return jsonify({"prediction": int(prediction), "probability": prediction_proba})
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Log the error with stack trace
        logger.error("Error during JSON prediction:", exc_info=True)
        return jsonify({"error": "An error occurred during prediction. Please check your inputs."}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# Run the application
if __name__ == '__main__':
    # Bind to all interfaces and use the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
