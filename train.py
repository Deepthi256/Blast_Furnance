# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    
    # Convert DATE_TIME to datetime object
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'], format='%d-%m-%y %H:%M')
    
    # Ensure all relevant columns are of float data type, replacing non-numeric values with NaN
    float_columns = data.columns.drop('DATE_TIME')
    data[float_columns] = data[float_columns].apply(pd.to_numeric, errors='coerce')
    
    # Resample data on an hourly basis and compute mean
    data.set_index('DATE_TIME', inplace=True)
    hourly_data = data.resample('h').mean()
    
    # Handle missing values by forward filling
    hourly_data = hourly_data.ffill()
    
    # Feature Engineering: Extract hour and day of the week from the datetime index
    hourly_data['HOUR'] = hourly_data.index.hour
    hourly_data['DAY_OF_WEEK'] = hourly_data.index.dayofweek
    hourly_data.reset_index(inplace=True)
    
    return hourly_data

def train_model(data):
    # Check if the target variable 'SKIN_TEMP_AVG' is in the dataset
    if 'SKIN_TEMP_AVG' not in data.columns:
        raise ValueError("The target variable 'SKIN_TEMP_AVG' is not in the dataset.")
    
    # Prepare feature matrix X and target vector y
    X = data.drop(columns=['DATE_TIME', 'SKIN_TEMP_AVG'])
    y = data['SKIN_TEMP_AVG']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    
    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    
    return model, X_train

if __name__ == '__main__':
    filepath = r"C:\Users\imkir\flask\bf-rinl.csv"  # Adjust this path if necessary
    data = load_and_preprocess_data(filepath)
    model, X_train = train_model(data)
    
    # Save the trained model to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the feature medians to a file
    feature_medians = X_train.median()
    feature_medians.to_pickle('feature_medians.pkl')
