from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained model and feature medians
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

feature_medians = pd.read_pickle('feature_medians.pkl')

def predict_skin_temp(model, input_features, feature_medians):
    # Ensure all columns used during training are present in the input_features dictionary
    feature_names = feature_medians.index
    input_data = {name: input_features.get(name, feature_medians[name]) for name in feature_names}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

@app.route('/')
def home():
    return render_template('furnance.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.json
    current_time = datetime.now()
    
    # Predict for the current time
    prediction_now = predict_skin_temp(model, input_features, feature_medians)
    
    # Prepare predictions for future hours
    predictions = {}
    hours = [1, 2, 3, 4, 5, 8, 10, 12, 20, 24]
    
    for hour in hours:
        future_time = current_time + timedelta(hours=hour)
        input_features['HOUR'] = future_time.hour
        prediction = predict_skin_temp(model, input_features, feature_medians)
        predictions[f'next_{hour}_hours'] = prediction
    
    return jsonify({
        'predicted_skin_temp_now': prediction_now,
        'predicted_skin_temp_next_1_hour': predictions['next_1_hours'],
        'predicted_skin_temp_next_2_hours': predictions['next_2_hours'],
        'predicted_skin_temp_next_3_hours': predictions['next_3_hours'],
        'predicted_skin_temp_next_4_hours': predictions['next_4_hours'],
        'predicted_skin_temp_next_5_hours': predictions['next_5_hours'],
        'predicted_skin_temp_next_8_hours': predictions['next_8_hours'],
        'predicted_skin_temp_next_10_hours': predictions['next_10_hours'],
        'predicted_skin_temp_next_12_hours': predictions['next_12_hours'],
        'predicted_skin_temp_next_20_hours': predictions['next_20_hours'],
        'predicted_skin_temp_next_24_hours': predictions['next_24_hours'],
        'date': current_time.strftime("%Y-%m-%d"),
        'hour': current_time.strftime("%H:%M:%S")
    })

if __name__ == '__main__':
    app.run(debug=True)
