from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
import mysql.connector
from sklearn.preprocessing import StandardScaler
import pymysql
# Database connection
mydb = pymysql.connect(
    host='localhost',
    port=3306,          
    user='root',        
    passwd='root',          
    database='phishing'  
)

mycur = mydb.cursor()
app = Flask(__name__)

# ======= Define the exact same features from your backend =======
# These should match the features selected by SelectKBest in your training
selected_features = [
    'length_url', 'length_hostname', 'ip', 'nb_qm', 'nb_eq', 'nb_slash',
    'nb_www', 'ratio_digits_url', 'ratio_digits_host', 'tld_in_subdomain',
    'prefix_suffix', 'shortest_word_host', 'longest_word_path',
    'phish_hints', 'nb_hyperlinks', 'ratio_intHyperlinks',
    'domain_in_title', 'domain_age', 'google_index', 'page_rank'
]

# ======= Load the Keras model =======
MODEL_PATH = 'best_keras_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found. Please ensure 'best_keras_model.h5' is in the current directory.")

model = load_model(MODEL_PATH)
print("✅ NAS Keras model loaded successfully!")
print(f"Features used: {len(selected_features)}")

# ======= Load your dataset to properly fit the scaler =======
def load_and_fit_scaler():
    """Load dataset and fit scaler exactly like in training"""
    try:
        # Load the same dataset you used for training
        df = pd.read_csv('dataset_B_05_2020 (1).csv')
        
        # Clean the data exactly as in training
        if 'url' in df.columns:
            df.drop('url', axis=1, inplace=True)
        
        # Convert 'status' to numeric binary values
        df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        numeric_imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(numeric_imputer.fit_transform(df), columns=df.columns)
        
        # Split features and target
        X = df_imputed.drop('status', axis=1)
        y = df_imputed['status']
        
        # Apply feature selection (SelectKBest) - same as training
        from sklearn.feature_selection import SelectKBest, f_classif
        k = 20
        selector = SelectKBest(score_func=f_classif, k=k)
        X_kbest = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_feature_names = X.columns[selector.get_support()].tolist()
        
        print(f"✅ Selected features from dataset: {selected_feature_names}")
        print(f"✅ Number of features: {len(selected_feature_names)}")
        
        # Fit StandardScaler on the selected features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[selected_feature_names])
        
        print("✅ Scaler fitted successfully on training data")
        
        return scaler, selected_feature_names
        
    except Exception as e:
        print(f"⚠️ Error loading dataset: {str(e)}")
        print("⚠️ Using manual initialization as fallback")
        
        # Fallback: Create a scaler with reasonable defaults
        scaler = StandardScaler()
        
        # Create dummy data to fit (using example values)
        dummy_data = []
        for i in range(len(selected_features)):
            # Create some variation around example values
            dummy_data.append([EXAMPLE_VALUES[i] - 1, EXAMPLE_VALUES[i] + 1])
        
        dummy_data = np.array(dummy_data).T
        scaler.fit(dummy_data)
        
        return scaler, selected_features

# ======= Initialize the scaler =======
scaler, selected_feature_names = load_and_fit_scaler()

# Update selected_features to match what was actually selected
selected_features = selected_feature_names
print(f"✅ Final feature list: {selected_features}")

# ======= Feature descriptions for the form =======
FEATURE_DESCRIPTIONS = {
    'length_url': 'Total length of the URL in characters',
    'length_hostname': 'Length of the hostname portion',
    'ip': 'Contains IP address (1=yes, 0=no)',
    'nb_qm': 'Number of question marks (?) in URL',
    'nb_eq': 'Number of equal signs (=) in URL',
    'nb_slash': 'Number of slashes (/) in URL',
    'nb_www': 'Contains www (1=yes, 0=no)',
    'ratio_digits_url': 'Ratio of digits in URL (0-1)',
    'ratio_digits_host': 'Ratio of digits in hostname (0-1)',
    'tld_in_subdomain': 'Top-level domain in subdomain (1=yes, 0=no)',
    'prefix_suffix': 'URL contains prefix/suffix (1=yes, 0=no)',
    'shortest_word_host': 'Length of shortest word in hostname',
    'longest_word_path': 'Length of longest word in path',
    'phish_hints': 'Number of phishing hints in URL',
    'nb_hyperlinks': 'Total number of hyperlinks in page',
    'ratio_intHyperlinks': 'Ratio of internal hyperlinks (0-1)',
    'domain_in_title': 'Domain appears in page title (1=yes, 0=no)',
    'domain_age': 'Age of domain in days',
    'google_index': 'Site indexed by Google (1=yes, 0=no)',
    'page_rank': 'PageRank of website (0-10)'
}

# ======= Default example values from your backend =======
EXAMPLE_VALUES = [77, 23, 1, 0, 0, 5, 0, 0.220779, 0.0, 0,
                  0, 19, 32, 0, 30, 0.966667, 1, 5767, 1, 2]

def predict_phishing(feature_values):
    """
    Predict if a URL is phishing or legitimate using the same steps as training
    
    Args:
        feature_values: List of feature values in exact order matching selected_features
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Convert to float
        feature_values = [float(x) for x in feature_values]
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([feature_values], columns=selected_features)
        
        print(f"✅ Raw input values: {feature_values}")
        
        # Scale the input using the fitted StandardScaler
        scaled_input = scaler.transform(input_df)
        
        print(f"✅ Scaled input shape: {scaled_input.shape}")
        
        # Get model predictions
        probs = model.predict(scaled_input, verbose=0)
        
        print(f"✅ Raw prediction probabilities: {probs}")
        
        # Extract probabilities
        legitimate_prob = probs[0][0]
        phishing_prob = probs[0][1]
        
        # Make prediction with 0.5 threshold (exact same as training)
        prediction = "Phishing" if phishing_prob > 0.5 else "Legitimate"
        
        return {
            'success': True,
            'prediction': prediction,
            'legitimate_probability': float(legitimate_prob),
            'phishing_probability': float(phishing_prob),
            'threshold': 0.5
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# ======= Routes =======
@app.route('/')
def index():
    """Home page with navigation"""
    return render_template('index.html')

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        address = request.form['address']
        
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password, Address) VALUES (%s, %s, %s, %s)'
                val = (name, email, password, address)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html') 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
                return redirect('/home')
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/models')
def models():
    """Models showcase page"""
    return render_template('models.html')

@app.route('/logout')
def logout():
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Main prediction page"""
    # GET request - show empty form
    if request.method == 'GET':
        return render_template('predict.html',
                             features=selected_features,
                             descriptions=FEATURE_DESCRIPTIONS,
                             example_values=EXAMPLE_VALUES,
                             form_data={},
                             url='')  # Empty URL for GET request
    
    # POST request
    try:
        # Get the URL (for display only, not used in prediction)
        url = request.form.get('url', '').strip()
        
        # Get form data
        feature_values = []
        errors = []
        
        # Extract all feature values in order
        for i, feature in enumerate(selected_features):
            value = request.form.get(f'feature_{i}', '').strip()
            if not value:
                errors.append(f"Feature '{feature}' is required")
            feature_values.append(value)
        
        if errors:
            return render_template('predict.html', 
                                 error="Please fill all fields",
                                 details=errors,
                                 features=selected_features,
                                 descriptions=FEATURE_DESCRIPTIONS,
                                 example_values=EXAMPLE_VALUES,
                                 form_data=request.form,
                                 url=url)  # Pass URL
        
        # Make prediction
        result = predict_phishing(feature_values)
        
        if not result['success']:
            return render_template('predict.html',
                                 error="Prediction failed",
                                 details=[result['error']],
                                 features=selected_features,
                                 descriptions=FEATURE_DESCRIPTIONS,
                                 example_values=EXAMPLE_VALUES,
                                 form_data=request.form,
                                 url=url)  # Pass URL
        
        # Format probabilities for display
        result['legitimate_probability_pct'] = f"{result['legitimate_probability']*100:.2f}%"
        result['phishing_probability_pct'] = f"{result['phishing_probability']*100:.2f}%"
        result['legitimate_probability'] = f"{result['legitimate_probability']:.4f}"
        result['phishing_probability'] = f"{result['phishing_probability']:.4f}"
        
        # Show the raw input values
        result['input_values'] = feature_values
        
        return render_template('predict.html', 
                             result=result,
                             features=selected_features,
                             descriptions=FEATURE_DESCRIPTIONS,
                             example_values=EXAMPLE_VALUES,
                             form_data=request.form,
                             url=url)  # Pass URL
        
    except Exception as e:
        # If url wasn't set in the try block
        url = request.form.get('url', '').strip() if 'url' in locals() else ''
        print(f"❌ Route error: {str(e)}")
        return render_template('predict.html',
                             error="An error occurred",
                             details=[str(e)],
                             features=selected_features,
                             descriptions=FEATURE_DESCRIPTIONS,
                             example_values=EXAMPLE_VALUES,
                             form_data=request.form,
                             url=url)  # Pass URL

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=5000)