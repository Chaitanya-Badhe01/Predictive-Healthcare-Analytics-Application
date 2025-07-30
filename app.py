import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, redirect, url_for, request, session, make_response, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Import LIME only if it's installed
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("LIME library not installed. LIME explanations will be disabled.")
    LIME_AVAILABLE = False

# Create an app object using the Flask class.
app = Flask(__name__)

# Create static directory if it doesn't exist
os.makedirs('static/images', exist_ok=True)

# Flask configurations
app.config['SECRET_KEY'] = 'rahul'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///merged_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF protection

# Initialize extensions
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Add zip to Jinja2 global environment so it can be used in templates
app.jinja_env.globals.update(zip=zip)

# Define the PyTorch model class for heart disease prediction
class HeartDiseaseModel(nn.Module):
    def __init__(self, in_features=18, h1=64, h2=28, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Load the trained models
try:
    print("Attempting to load diabetes model...")
    diabetes_model = joblib.load('diabetes_model.pkl')
    print(f"Diabetes model loaded successfully: {type(diabetes_model)}")
except Exception as e:
    print(f"Error loading diabetes model: {e}")
    diabetes_model = None

# Load the PyTorch heart disease model
try:
    print("Attempting to load heart model...")
    heart_model = HeartDiseaseModel()
    heart_model.load_state_dict(torch.load('heart_model.pth'))
    heart_model.eval()  # Important! Set to evaluation mode
    print("Heart model loaded successfully")
except Exception as e:
    print(f"Error loading PyTorch heart model: {e}")
    heart_model = None

# Load the actual training data for LIME explanations
try:
    heart_training_data = np.load('heart_training_data.npy')
    print(f"Loaded heart training data with shape: {heart_training_data.shape}")
except Exception as e:
    print(f"Error loading heart training data: {e}")
    heart_training_data = None

# Try to load diabetes training data if available
try:
    diabetes_training_data = np.load('diabetes_training_data.npy')
    print(f"Loaded diabetes training data with shape: {diabetes_training_data.shape}")
except Exception as e:
    print(f"Diabetes training data not found: {e}")
    diabetes_training_data = None

# Feature names for LIME explanation
heart_feature_names = [
    'Chest Pain', 'Shortness of Breath', 'Fatigue', 'Palpitations', 
    'Dizziness', 'Swelling', 'Pain Arms/Jaw/Back', 'Cold Sweats/Nausea',
    'High Blood Pressure', 'High Cholesterol', 'Diabetes', 'Smoking',
    'Obesity', 'Sedentary Lifestyle', 'Family History', 'Chronic Stress',
    'Gender (Male=1)', 'Age'
]

# Feature names for Diabetes LIME explanation
diabetes_feature_names = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden Weight Loss',
    'Weakness', 'Polyphagia', 'Genital Thrush', 'Visual Blurring',
    'Itching', 'Irritability', 'Delayed Healing', 'Partial Paresis',
    'Muscle Stiffness', 'Alopecia', 'Obesity'
]

# DATABASE MODELS
# User model for login system
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    is_admin = db.Column(db.Boolean, default=False)  # Add admin flag

# Admin Login Form
class AdminLoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Default admin credentials
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "secure123"  # Use a strong password in production
# Patient profile model
class PatientProfile(db.Model):
    sno = db.Column(db.Integer, primary_key=True)   
    name = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    Age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    height = db.Column(db.Integer, nullable=False)
    Weight = db.Column(db.Integer, nullable=False)
    bloodgroup = db.Column(db.String(50), nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno} {self.name} {self.phone} {self.email} {self.Age} {self.gender} {self.height} {self.Weight} {self.bloodgroup}"

# Appointment model
class Appointment(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    appoinmentname = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    doctor = db.Column(db.String(80), nullable=False)
    message = db.Column(db.String(250), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"{self.sno} {self.appoinmentname} {self.email} {self.doctor} {self.message}"

# Contact model
class Contact(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    message = db.Column(db.String(250), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self) -> str:
        return f"{self.sno} {self.name} {self.email} {self.message}"

# Contact Form
class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    message = StringField('Message', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Patient Profile Form
class PatientProfileForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    phone = StringField('Phone', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    Age = StringField('Age', validators=[DataRequired()])
    gender = StringField('Gender', validators=[DataRequired()])
    height = StringField('Height', validators=[DataRequired()])
    Weight = StringField('Weight', validators=[DataRequired()])
    bloodgroup = StringField('Blood Group', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Appointment Form
class AppointmentForm(FlaskForm):
    appoinmentname = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    doctor = StringField('Doctor', validators=[DataRequired()])
    message = StringField('Message', validators=[DataRequired()])
    submit = SubmitField('Submit')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms for login and signup
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)]) 
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    submit = SubmitField('Register')

# Heart Disease Prediction Form
class HeartPredictionForm(FlaskForm):
    # Add fields for heart disease prediction
    # For each question in your heart_question.html form
    feature1 = StringField('Feature 1', validators=[DataRequired()])
    feature2 = StringField('Feature 2', validators=[DataRequired()])
    # Add all other features as needed
    submit = SubmitField('Predict')

# Diabetes Prediction Form
class DiabetesPredictionForm(FlaskForm):
    # Add fields for diabetes prediction
    # For each question in your questions.html form
    feature1 = StringField('Feature 1', validators=[DataRequired()])
    feature2 = StringField('Feature 2', validators=[DataRequired()])
    # Add all other features as needed
    submit = SubmitField('Predict')

# HOME AND MAIN ROUTES
# Home page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/technology')
def technology():
    return render_template("technology.html")

# About page
@app.route('/about')
def about():
    return render_template("about.html")

# Help page
@app.route('/help')
def help():
    return render_template("help.html")

# Terms and conditions page
@app.route('/tc')
def terms():
    return render_template("tc.html")

# Contact page
@app.route('/contact', methods=["GET", "POST"])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        contact_entry = Contact(
            name=form.name.data,
            email=form.email.data,
            message=form.message.data
        )
        db.session.add(contact_entry)
        db.session.commit()
        return render_template('return3.html')
    return render_template('contact.html', form=form)

# AUTHENTICATION ROUTES
# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
        return render_template("login.html", form=form, error="Invalid username or password")
    return render_template("login.html", form=form)

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")
    return render_template('signup.html', form=form)

# Logout route
@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# DASHBOARD AND PATIENT PROFILE
# Dashboard route
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

# Patient profile form
@app.route('/form', methods=["GET", "POST"])
@login_required
def patient_profile():
    form = PatientProfileForm()
    if form.validate_on_submit():
        try:
            patient = PatientProfile(
                name=form.name.data,
                phone=form.phone.data,
                email=form.email.data,
                Age=int(form.Age.data),
                gender=form.gender.data,
                height=int(form.height.data),
                Weight=int(form.Weight.data),
                bloodgroup=form.bloodgroup.data
            )
            
            db.session.add(patient)
            db.session.commit()
            return render_template('return.html')
        except Exception as e:
            print(f"Error processing form: {e}")
            db.session.rollback()  # Roll back any failed transaction
            return render_template('form.html', form=form, error="There was an error processing your form. Please try again.")
    
    return render_template('form.html', form=form)

# DOCTOR CONSULTATION ROUTES
# Doctor listing
@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

# Doctor info pages
@app.route('/doctinfo1')
def doctinfo1():
    return render_template('doctinfo1.html')

@app.route('/doctinfo2')
def doctinfo2():
    return render_template('doctinfo2.html')

@app.route('/doctinfo3')
def doctinfo3():
    return render_template('doctinfo3.html')

# Appointment booking
@app.route('/appointment', methods=["GET", "POST"])
@login_required
def appointment():
    form = AppointmentForm()
    if form.validate_on_submit():
        try:
            booking = Appointment(
                appoinmentname=form.appoinmentname.data,
                email=form.email.data,
                doctor=form.doctor.data,
                message=form.message.data
            )
            
            db.session.add(booking)
            db.session.commit()
            return render_template('return2.html')
        except Exception as e:
            print(f"Error booking appointment: {e}")
            db.session.rollback()
            return render_template('appoinment.html', form=form, error="There was an error booking your appointment. Please try again.")
    
    return render_template('appoinment.html', form=form)

# For backward compatibility
@app.route('/appoinment', methods=["GET", "POST"])
def appoinment():
    return appointment()

# Video consultation
@app.route('/videocall')
@login_required
def videocall():
    return render_template('videocall.html')

@app.route('/gmeet')
@login_required
def gmeet():
    return render_template('gmeet.html')

# DISEASE PREDICTION ROUTES
# Disease index
@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

# Diabetes Questions page
@app.route('/questions')
def questions():
    return render_template('questions.html')

# Heart Question page
@app.route('/heart_question')
def heart_question():
    return render_template('heart_question.html')

# New Heart Question page
@app.route('/new_heart_question')
def new_heart_question():
    return render_template('new_heart_question.html')

# Define prediction functions for LIME explanations
def heart_predict_fn(input_data):
    """Function for LIME to use when explaining heart disease predictions"""
    if len(input_data.shape) == 1:
        input_data = np.array([input_data])
    
    # Convert to tensor for PyTorch model
    input_tensor = torch.FloatTensor(input_data)
    
    # Get probability from the model
    with torch.no_grad():
        outputs = heart_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).numpy()
    
    return probabilities

def diabetes_predict_fn(input_data):
    """Function for LIME to use when explaining diabetes predictions"""
    if len(input_data.shape) == 1:
        input_data = np.array([input_data])
    # Get probability estimates from the diabetes model
    probabilities = diabetes_model.predict_proba(input_data)
    return probabilities

@app.route("/heart", methods=['GET', 'POST'])
@login_required
def heart_predict():
    if request.method == 'POST':
        try:
            # Collect form data
            form_values = request.form.values()
            int_features = [int(x) for x in form_values]
            features = np.array([int_features])
            
            # Check if model is loaded
            if heart_model is None:
                return render_template("heart_results.html", error="Heart disease prediction model not loaded. Please contact support.")
            
            # Convert to tensor for PyTorch model
            features_tensor = torch.FloatTensor(features)
            
            # Make prediction
            try:
                with torch.no_grad():
                    outputs = heart_model(features_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probabilities = F.softmax(outputs, dim=1)
                
                result = "Positive" if predicted.item() == 1 else "Negative"
                probability = f"{probabilities[0][predicted.item()].item()*100:.2f}%"
            except Exception as model_error:
                print(f"Error during model prediction: {model_error}")
                import traceback
                traceback.print_exc()
                return render_template("heart_results.html", error=f"Error during model prediction: {model_error}")
            
            # Feature importance and explanation image
            feature_importances = None
            explanation_image = None
            
            # Try to generate LIME explanation if available
            if LIME_AVAILABLE and heart_training_data is not None:
                try:
                    # Setup LIME explainer
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        heart_training_data,
                        feature_names=heart_feature_names,
                        class_names=['Negative', 'Positive'],
                        discretize_continuous=True,
                        mode='classification'
                    )
                    
                    # Generate explanation
                    exp = explainer.explain_instance(
                        np.array(int_features), 
                        heart_predict_fn, 
                        num_features=len(heart_feature_names)
                    )
                    
                    # Save explanation visualization
                    img_filename = f'heart_explanation_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
                    exp.save_to_file(os.path.join('static/images', img_filename))
                    explanation_image = f'images/{img_filename}'
                    
                    # Get feature importances
                    feature_importances = exp.as_list()
                except Exception as lime_error:
                    print(f"Error generating LIME explanation: {lime_error}")
                    import traceback
                    traceback.print_exc()
            
            # Pass all variables to the template
            return render_template(
                'heart_results.html',
                prediction=result,
                probability=probability,
                disease_type="Heart Disease",
                feature_importances=feature_importances,
                explanation_image=explanation_image,
                LIME_AVAILABLE=LIME_AVAILABLE
            )
            
        except Exception as e:
            print(f"Unhandled error in heart disease prediction: {e}")
            import traceback
            traceback.print_exc()
            return render_template('heart_results.html', error=f"An unexpected error occurred: {str(e)}")
    
    # On GET request, return the form
    return render_template('heart_question.html')

@app.route("/diabetes", methods=['GET', 'POST'])
@login_required
def diabetes():
    if request.method == 'POST':
        try:
            # Collect form data
            form_values = request.form.values()
            int_features = [int(x) for x in form_values]
            features = np.array([int_features])
            
            if diabetes_model is None:
                return render_template("results.html", error="Diabetes prediction model not loaded.")

            # Get prediction with better error handling
            try:
                prediction = diabetes_model.predict(features)
                result = prediction[0]
                
                # Get probability
                proba = diabetes_model.predict_proba(features)[0]
                
                # Make sure probabilities are aligned correctly
                probability = f"{proba[1]*100:.2f}%" if result == "Positive" else f"{proba[0]*100:.2f}%"
            except Exception as model_error:
                print(f"Error in model prediction: {model_error}")
                return render_template("results.html", error=f"Model prediction error: {model_error}")
            
            # Feature importance and explanation image
            feature_importances = None
            explanation_image = None
            
            # Debug message
            print("Attempting to generate LIME explanation...")
            
            # Try to generate LIME explanation
            try:
                # Import LIME with debug message
                print("Importing LIME packages...")
                import lime
                import lime.lime_tabular
                print("LIME successfully imported")
                
                # Debug checks for required variables
                if diabetes_training_data is None:
                    print("ERROR: diabetes_training_data is None")
                    return render_template("results.html", prediction=result, probability=probability, 
                                          disease_type="Diabetes", error="Missing training data for LIME explanation")
                
                if diabetes_feature_names is None:
                    print("ERROR: diabetes_feature_names is None")
                    return render_template("results.html", prediction=result, probability=probability, 
                                          disease_type="Diabetes", error="Missing feature names for LIME explanation")
                
                print(f"Training data shape: {np.array(diabetes_training_data).shape}")
                print(f"Feature names: {diabetes_feature_names}")
                print(f"Number of features in input: {len(int_features)}")
                print(f"Number of feature names: {len(diabetes_feature_names)}")
                
                # Check if feature count matches
                if len(int_features) != len(diabetes_feature_names):
                    print(f"ERROR: Feature count mismatch! Input has {len(int_features)} features but {len(diabetes_feature_names)} feature names defined")
                    return render_template("results.html", prediction=result, probability=probability, 
                                          disease_type="Diabetes", error="Feature count mismatch for LIME explanation")
                
                # Create directory for images
                img_dir = 'static/images'
                os.makedirs(img_dir, exist_ok=True)
                print(f"Image directory created/verified: {img_dir}")
                
                # Define the prediction function
                def diabetes_predict_fn(x):
                    return diabetes_model.predict_proba(x)
                
                print("Setting up LIME explainer...")
                
                # Convert training data to numpy array if not already
                training_data_array = np.array(diabetes_training_data)
                
                # Setup LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=training_data_array,
                    feature_names=diabetes_feature_names,
                    class_names=['Negative', 'Positive'],
                    discretize_continuous=True,
                    mode='classification'
                )
                
                print("LIME explainer created successfully")
                print("Generating explanation instance...")
                
                # Generate explanation
                exp = explainer.explain_instance(
                    data_row=np.array(int_features).astype(float), 
                    predict_fn=diabetes_predict_fn, 
                    num_features=len(diabetes_feature_names)
                )
                
                print("Explanation generated successfully")
                
                # Save explanation visualization
                img_filename = f'diabetes_explanation_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
                img_path = os.path.join(img_dir, img_filename)
                print(f"Saving explanation to: {img_path}")
                
                # Use matplotlib directly to save the figure
                plt.figure(figsize=(10, 6))
                exp.as_pyplot_figure()
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close()
                
                print(f"Explanation image saved at: {img_path}")
                
                # Set the image path for the template
                explanation_image = f'images/{img_filename}'
                
                # Get feature importances
                feature_importances = exp.as_list()
                print(f"Feature importances: {feature_importances}")
                
            except ImportError as ie:
                print(f"LIME import error: {ie}")
                return render_template("results.html", prediction=result, probability=probability, 
                                      disease_type="Diabetes", error="LIME package not installed")
            except Exception as lime_error:
                print(f"Error generating LIME explanation: {lime_error}")
                import traceback
                traceback.print_exc()
                return render_template("results.html", prediction=result, probability=probability, 
                                      disease_type="Diabetes", error=f"LIME error: {str(lime_error)}")
            
            # Check if we have LIME results
            if explanation_image is None:
                print("WARNING: No explanation image was generated")
            
            print(f"Rendering template with explanation_image={explanation_image}")
            
            return render_template(
                'results.html',
                prediction=result,
                probability=probability,
                disease_type="Diabetes",
                feature_importances=feature_importances,
                explanation_image=explanation_image
            )
            
        except Exception as e:
            print(f"Error in diabetes prediction: {e}")
            import traceback
            traceback.print_exc()
            return render_template('results.html', error=f"An error occurred: {str(e)}")
    
    # On GET request, return the form
    return render_template('questions.html')

# New heart disease prediction route
@app.route("/new_heart_predict", methods=['GET', 'POST'])
@login_required
def new_heart_predict():
    if request.method == 'POST':
        try:
            # Collect form data
            form_values = request.form.values()
            int_features = [int(x) for x in form_values]
            features = np.array([int_features])
            
            if heart_model is None:
                return render_template("heart_results.html", error="Heart disease prediction model not loaded.")
            
            # Convert to tensor for PyTorch model
            features_tensor = torch.FloatTensor(features)
            
            # Get prediction
            with torch.no_grad():
                outputs = heart_model(features_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = F.softmax(outputs, dim=1)
            
            result = "Positive" if predicted.item() == 1 else "Negative"
            probability = f"{probabilities[0][predicted.item()].item()*100:.2f}%"
            
            # Feature importance and explanation image
            feature_importances = None
            explanation_image = None
            
            # Try to generate LIME explanation
            if LIME_AVAILABLE and heart_training_data is not None:
                try:
                    # Set up LIME explainer
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        heart_training_data,
                        feature_names=heart_feature_names,
                        class_names=['Negative', 'Positive'],
                        discretize_continuous=True,
                        mode='classification'
                    )
                    
                    # Generate explanation
                    exp = explainer.explain_instance(
                        np.array(int_features), 
                        heart_predict_fn, 
                        num_features=len(heart_feature_names)
                    )
                    
                    # Save explanation image
                    img_filename = f'new_heart_explanation_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
                    exp.save_to_file(os.path.join('static/images', img_filename))
                    explanation_image = f'images/{img_filename}'
                    
                    # Get feature importance
                    feature_importances = exp.as_list()
                except Exception as e:
                    print(f"Error generating LIME explanation: {e}")
                    explanation_image = None
                    feature_importances = None
            
            return render_template(
                'heart_results.html',
                prediction=result,
                probability=probability,
                disease_type="Heart Disease",
                feature_importances=feature_importances,
                explanation_image=explanation_image
            )
            
        except Exception as e:
            print(f"Error in new heart disease prediction: {e}")
            import traceback
            traceback.print_exc()
            return render_template('heart_results.html', error="An error occurred during prediction.")
    
    # On GET request, return the form
    return render_template('new_heart_question.html')

# API ROUTES
# API for getting user profile
@app.route('/api/profile', methods=['GET'])
@login_required
def api_profile():
    try:
        # Find patient profile by email
        profile = PatientProfile.query.filter_by(email=current_user.email).first()
        if not profile:
            return {'error': 'Profile not found'}, 404
        
        return {
            'name': profile.name,
            'email': profile.email,
            'phone': profile.phone,
            'age': profile.Age,
            'gender': profile.gender,
            'height': profile.height,
            'weight': profile.Weight,
            'bloodgroup': profile.bloodgroup
        }
    except Exception as e:
        print(f"API error: {e}")
        return {'error': str(e)}, 500

# API for appointment history
@app.route('/api/appointments', methods=['GET'])
@login_required
def api_appointments():
    try:
        appointments = Appointment.query.filter_by(email=current_user.email).all()
        result = []
        
        for apt in appointments:
            result.append({
                'id': apt.sno,
                'name': apt.appoinmentname,
                'doctor': apt.doctor,
                'message': apt.message,
                'date': apt.date.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return {'appointments': result}
    except Exception as e:
        print(f"API error: {e}")
        return {'error': str(e)}, 500

# ADMIN ROUTES
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if username == DEFAULT_ADMIN_USERNAME and password == DEFAULT_ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            error = "Invalid credentials. Please try again."
            return render_template("admin_login.html", error=error)
    
    return render_template("admin_login.html")

@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    
    return render_template("admin_dashboard.html")

# View all patients
@app.route('/admin_patients')
@login_required
def admin_patients():
    if not current_user.is_admin:
        flash('Access denied: Admin privileges required')
        return redirect(url_for('dashboard'))
    
    patients = PatientProfile.query.all()
    return render_template('admin_patients.html', patients=patients)

# View all appointments
@app.route('/admin_appointments')
@login_required
def admin_appointments():
    if not current_user.is_admin:
        flash('Access denied: Admin privileges required')
        return redirect(url_for('dashboard'))
    
    appointments = Appointment.query.all()
    return render_template('admin_appointments.html', appointments=appointments)

# View all contacts/inquiries
@app.route('/admin_contacts')
@login_required
def admin_contacts():
    if not current_user.is_admin:
        flash('Access denied: Admin privileges required')
        return redirect(url_for('dashboard'))
    
    contacts = Contact.query.all()
    return render_template('admin_contacts.html', contacts=contacts)

# Create admin user route
@app.route('/create_admin', methods=['GET', 'POST'])
def create_admin():
    # Check if admin already exists
    if User.query.filter_by(is_admin=True).first():
        return "Admin already exists"
    
    # Create admin if not exists
    hashed_password = generate_password_hash('admin_password', method='pbkdf2:sha256')
    admin = User(username='admin', email='admin@example.com', password=hashed_password, is_admin=True)
    db.session.add(admin)
    db.session.commit()
    return "Admin created successfully"

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Create all database tables if they don't exist
with app.app_context():
    db.create_all()

# Run the app
if __name__ == '__main__':
    app.run(debug=True)