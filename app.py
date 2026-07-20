import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM Model
MODEL_PATH = "SVMModel.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Custom Minimalist Dark Theme UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            text-align: center;
            letter-spacing: -0.025em;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 2rem;
        }

        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }

        .full-width {
            grid-column: span 2;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input, select {
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
            outline: none;
            width: 100%;
        }

        input:focus, select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            padding: 0.85rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 1rem;
        }

        button:hover {
            background-color: var(--accent-hover);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            border: 1px solid transparent;
        }

        .result-success {
            background-color: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            color: #34d399;
        }

        .result-fail {
            background-color: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.3);
            color: #f87171;
        }

        @media (max-width: 600px) {
            form {
                grid-template-columns: 1fr;
            }
            .full-width {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Predictive Analytics Model</h1>
    <p class="subtitle">Enter student metrics to compute SVM classification</p>

    <form action="/predict" method="POST">
        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="0">Male</option>
                <option value="1">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" min="0" max="100" placeholder="e.g. 21" required>
        </div>

        <div class="form-group">
            <label for="study_hours">Study Hours / Week</label>
            <input type="number" id="study_hours" name="study_hours" step="0.1" placeholder="e.g. 15.5" required>
        </div>

        <div class="form-group">
            <label for="attendance">Attendance Rate (%)</label>
            <input type="number" id="attendance" name="attendance" step="0.1" min="0" max="100" placeholder="e.g. 92.5" required>
        </div>

        <div class="form-group">
            <label for="parent_education">Parent Education Level</label>
            <select id="parent_education" name="parent_education" required>
                <option value="0">High School</option>
                <option value="1">Associate's Degree</option>
                <option value="2">Bachelor's Degree</option>
                <option value="3">Master's / Ph.D.</option>
            </select>
        </div>

        <div class="form-group">
            <label for="internet">Internet Access</label>
            <select id="internet" name="internet" required>
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>

        <div class="form-group">
            <label for="extracurricular">Extracurricular Activities</label>
            <select id="extracurricular" name="extracurricular" required>
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>

        <div class="form-group">
            <label for="previous_score">Previous Score</label>
            <input type="number" id="previous_score" name="previous_score" step="0.1" placeholder="e.g. 78.0" required>
        </div>

        <div class="form-group full-width">
            <label for="final_score">Final Term Score</label>
            <input type="number" id="final_score" name="final_score" step="0.1" placeholder="e.g. 82.5" required>
        </div>

        <button type="submit" class="full-width">Run Model Assessment</button>
    </form>

    {% if prediction %}
        <div class="result-box {% if prediction == 'Yes' %}result-success{% else %}result-fail{% endif %}">
            Classification Result: {{ prediction }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template_string(HTML_TEMPLATE, prediction="Error: Model file missing.")
    
    try:
        # Extract and parse values exactly matching the trained numerical format
        features = [
            float(request.form["gender"]),
            float(request.form["age"]),
            float(request.form["study_hours"]),
            float(request.form["attendance"]),
            float(request.form["parent_education"]),
            float(request.form["internet"]),
            float(request.form["extracurricular"]),
            float(request.form["previous_score"]),
            float(request.form["final_score"])
        ]
        
        # Convert into a 2D array format for standard scikit-learn input
        input_data = np.array([features])
        
        # Make the prediction
        prediction_val = model.predict(input_data)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=str(prediction_val))
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Execution failed: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
