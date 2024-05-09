import pickle
import secrets
from flask import Flask, request, app, jsonify, url_for, render_template, redirect
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from flask_bootstrap import Bootstrap5
from datetime import datetime

from flask_wtf import FlaskForm, CSRFProtect
from wtforms import  StringField, IntegerField, SelectField, SubmitField,FileField
from wtforms.validators import DataRequired, Length,  InputRequired
from flask import flash
from flask import session


app = Flask(__name__)

foo = secrets.token_urlsafe(16)
app.secret_key = foo
# Bootstrap-Flask requires this line
bootstrap = Bootstrap5(app)
# Flask-WTF requires this line
csrf = CSRFProtect(app)

app.config['UPLOAD_FOLDER'] = 'input'

regmodel = pickle.load(open('financial_inclusion_reg_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class BankAccountForm(FlaskForm):
    country = StringField('Country', validators=[DataRequired()])
    year = IntegerField('Year', validators=[DataRequired()])
    uniqueid = StringField('Unique ID', validators=[DataRequired()])
    location_type = SelectField('Location Type', choices=[('Rural', 'Rural'), ('Urban', 'Urban')], validators=[DataRequired()])
    cellphone_access = SelectField('Cellphone Access', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    household_size = IntegerField('Household Size', validators=[DataRequired()])
    age_of_respondent = IntegerField('Age of Respondent', validators=[DataRequired()])
    gender_of_respondent = SelectField('Gender of Respondent', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    relationship_with_head = StringField('Relationship with Head', validators=[DataRequired()])
    marital_status = StringField('Marital Status', validators=[DataRequired()])
    education_level = StringField('Education Level', validators=[DataRequired()])
    job_type = StringField('Job Type', validators=[DataRequired()])
    submit = SubmitField('Submit')


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/')
def home():
    return render_template('home.html')


# Load the LabelEncoder
le = LabelEncoder()


def preprocessing_data(data):
    data = data.drop(['uniqueid'], axis=1)
    # Convert the following numerical labels from integer to float
    float_array = data[["household_size", "age_of_respondent"]].values.astype(float)
    
    # Categorical features to be converted to One Hot Encoding
    categ = ["relationship_with_head", "marital_status", "education_level", "job_type", "country"]
    
    # One Hot Encoding conversion
    data = pd.get_dummies(data, prefix_sep="_", columns=categ)

    data = data.reindex(columns=model_columns, fill_value=0)
    
    # Convert boolean columns to integers (1 and 0)
    boolean_columns = ["relationship_with_head_Other non-relatives", "job_type_Government Dependent", 
                       "job_type_Informally employed", "job_type_No Income", "job_type_Other Income",
                       "job_type_Remittance Dependent", "job_type_Self employed", "country_Kenya",
                       "country_Rwanda", "country_Tanzania", "country_Uganda"]
    
    # Convert boolean columns to integers
    data[boolean_columns] = data[boolean_columns].astype(int)
    
    # Convert remaining boolean columns to integers
    boolean_columns_remaining = [col for col in data.columns if data[col].dtype == bool]
    data[boolean_columns_remaining] = data[boolean_columns_remaining].astype(int)
    
    # Convert remaining categorical columns to numerical using Label Encoder
    data["location_type"] = le.fit_transform(data["location_type"])
    data["cellphone_access"] = le.fit_transform(data["cellphone_access"])
    data["gender_of_respondent"] = le.fit_transform(data["gender_of_respondent"])
    
    return data

def extract_form_data(form):
     data = {
        "country": form.country.data,
        "year": form.year.data,
        "uniqueid": form.uniqueid.data,
        "location_type": form.location_type.data,
        "cellphone_access": form.cellphone_access.data,
        "household_size": form.household_size.data,
        "age_of_respondent": form.age_of_respondent.data,
        "gender_of_respondent": form.gender_of_respondent.data,
        "relationship_with_head": form.relationship_with_head.data,
        "marital_status": form.marital_status.data,
        "education_level": form.education_level.data,
        "job_type": form.job_type.data
    }
     return data

csrf = CSRFProtect(app)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    data = pd.DataFrame(data)
    print("Normal Data")
    print(data.info())
    test1 = preprocessing_data(data)
    print()
    print("Dumped data")
    print(test1.info())
    data_scaled = scaler.transform(test1)
    prediction = regmodel.predict(data_scaled)
    print(prediction[0])

    prediction_list = prediction.tolist()
    return jsonify(prediction_list)

@app.route('/individual', methods=['GET','POST'])
def predict_individual():
    form = BankAccountForm()
    message = ""
    if form.validate_on_submit():
        # Process the form data here
        data = extract_form_data(form)
        print(data)

        data_df = pd.DataFrame([data.values()], columns=data.keys())
        
    
        preporocessed_data = preprocessing_data(data_df)

        print(preporocessed_data)

        data_scaled = scaler.transform(preporocessed_data)

        prediction = regmodel.predict(data_scaled)

        if prediction == 1:
            print("You qualify to open a bank account")
        elif prediction == 0:
            print("You do not qualify to open a bank account")

        prediction_message = "Congratulations 🎉, you qualify to open a bank account" if prediction == 1 else "Sorry 🙁, do not qualify to open a bank account"

        print(f"The prediction is likely to be {prediction}")

        return  redirect(url_for('result', prediction_message=prediction_message))
    return render_template('individual_prediction.html', form=form)


@app.route('/organisation', methods=['GET','POST'])
def predict_organisation():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        if file and allowed_file(file.filename):
            
            # file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)


            df = pd.read_csv(file_path)
            preporocessed_data = preprocessing_data(df)

            print(preporocessed_data)

            data_scaled = scaler.transform(preporocessed_data)

            prediction = regmodel.predict(data_scaled)

            print(f"The prediction is likely to be {prediction}")

            new_dataset = create_predicted_dataset(prediction, df)

            # Calculate location distribution
            
            distributions = [
                display_location_distribution(new_dataset),
                display_country_distribution(new_dataset),
                display_cellphone_access_distribution(new_dataset),
                display_marital_relationship_distribution(new_dataset),
                display_job_income_distribution(new_dataset)
            ]
            
           # Convert distributions to string format
            distribution_strings = ['&'.join([f"{key}={value}" for key, value in dist.items()]) for dist in distributions]

            # Concatenate distribution strings
            all_distributions_str = ','.join(distribution_strings)
            

            return redirect(url_for('organisation_result',all_distributions_str=all_distributions_str))
        
        else:
            flash("Only CSV files are allowed.")
            return "File has been declined."
    return render_template('organisational_prediction.html', form=form)

def create_predicted_dataset(prediction_score, df):
    predicted_labels_df = pd.DataFrame(prediction_score, columns = ['predicted_label'])
    merged_dataset = pd.concat([df, predicted_labels_df], axis=1)
    return merged_dataset

#location
def display_location_distribution(dataset):
     location_distribution = dataset[dataset['predicted_label'] == 1]['location_type'].value_counts(normalize=True)
     location_dict = location_distribution.to_dict()
     return location_dict

#country
def display_country_distribution(dataset):
    country_distribution = dataset[dataset['predicted_label'] == 1]['country'].value_counts(normalize=True)
    country_distribution = country_distribution.to_dict()
    return country_distribution

#Access to Cellphone  Services
def display_cellphone_access_distribution(dataset):
    access_distribution = dataset[dataset['predicted_label'] == 1]['cellphone_access'].value_counts(normalize=True)
    access_dict = access_distribution.to_dict()
    return access_dict

#Marital Status 
def display_marital_relationship_distribution(dataset):
    marital_relationship_distribution = dataset[dataset['predicted_label'] == 1]['marital_status'].value_counts(normalize=True)
    marital_relationship_dict = marital_relationship_distribution.to_dict()
    return marital_relationship_dict

#Job Type
def display_job_income_distribution(dataset):
    job_income_distribution = dataset[dataset['predicted_label'] == 1]['job_type'].value_counts(normalize=True)
    job_income_dict = job_income_distribution.to_dict()
    return job_income_dict

@app.route('/result')
def result():
    prediction_message = request.args.get('prediction_message')
    return render_template('result.html', prediction_message=prediction_message)

# @app.route('/organisation_result')
# def organisation_result():
#     distributions_str = request.args.get('distributions')
#     distributions = distributions_str.split(',')
#     return render_template('organisation_result.html', distributions=distributions )

@app.route('/organisation_result')
def organisation_result():
    distributions_str = request.args.get('all_distributions_str')
    
    # Split the string into key-value pairs
    distributions_list = distributions_str.split('&')
    
    # Create a dictionary to store the distributions
    distributions = {}
    for item in distributions_list:
        parts = item.split('=')
        if len(parts) == 2:
            key, value = parts
            distributions[key] = float(value) * 100  # Convert value to percentage
        else:
            print(f"Ignoring malformed distribution: {item}")
    
    # Format the distributions
    formatted_distributions = [f"{key}: {value:.2f}%" for key, value in distributions.items()]
    
    return render_template('organisation_result.html', distributions=formatted_distributions)



    

    
if __name__ == "__main__":
    app.run(debug=True)
