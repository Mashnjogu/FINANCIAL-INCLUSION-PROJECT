import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
regmodel = pickle.load(open('financial_inclusion_reg_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

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
    return jsonify(prediction_list[0])
    



    
if __name__ == "__main__":
    app.run(debug=True)
