from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
from application_logging.logger import App_Logger
from flask import Response
import pickle
import pandas as pd
from data_ingestion.data_loader import Data_Getter
import numpy as np
from File_Operation.FileOperation import File_Operations
from Functions.functions import convert_sqft_to_num
import json
from ModelTraining.trainingModel import trainModel

app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # load the model
        logger = App_Logger()
        file_prediction = open("Logs/Prediction_Log.txt", 'a+')
        file_io = File_Operations(logger_object=logger, file_object=file_prediction)
        model_pipeline = file_io.load_model('RandomForestRegressor')

        # load encoded_features dictionary
        with open('encoded_features.json', 'r') as myfile:
            encoded_features_str = myfile.read()
            encoded_features = json.loads(encoded_features_str)

        # get user input from html form
        total_sqft_form = request.form['total_sqft']
        balcony_form = request.form['balcony']
        society_form = request.form['society']
        area_type_form = request.form['area_type']
        location_form = request.form['location']

        total_sqft = convert_sqft_to_num(total_sqft_form)
        if total_sqft != 0:
            total_sqft_list = [np.log(total_sqft)]
        else:
            total_sqft += 0.00000000000000000000000000001
            total_sqft_list = [np.log(total_sqft)]


        # preprocess balcony (no need since it will be in numerical format)
        balcony_list = [balcony_form]

        # preprocess society
        new_society = 'society_' + society_form
        encoded_society = encoded_features['society']
        society_vec = np.zeros(len(encoded_society))
        for i in range(len(encoded_society)):
            if encoded_society[i] == new_society:
                society_vec[i] = 1

        society_list = list(society_vec)

        # preprocess area_type
        if area_type_form == 'Super built-up Area':
            area_type_Superbuiltup = 1
            area_type_Carpet = 0
            area_type_Plot = 0
        elif area_type_form == 'Plot Area':
            area_type_Superbuiltup = 0
            area_type_Carpet = 0
            area_type_Plot = 1
        elif area_type_form == 'Built-up Area':
            area_type_Superbuiltup = 0
            area_type_Carpet = 0
            area_type_Plot = 0

        elif area_type_form == 'Carpet Area':
            area_type_Superbuiltup = 0
            area_type_Carpet = 1
            area_type_Plot = 0

        area_type_list = [area_type_Carpet, area_type_Plot, area_type_Superbuiltup]

        # preprocess location
        encoded_location = encoded_features['location']
        location_vec = np.zeros(len(encoded_location))
        for i in range(len(encoded_location)):
            if encoded_location[i] == location_form:
                location_vec[i] = 1

        location_list = list(location_vec)

        x = total_sqft_list + balcony_list + society_list + area_type_list + location_list
        pred = model_pipeline.predict([x])

        return render_template('home.html', prediction_text='The House Price is : {}'.format(pred[0]))

    return render_template("home.html")

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json['folderTrainingPath'] is not None:
            training_file_path = request.json['folderTrainingPath']

            trainModelObj = trainModel(training_file_path=training_file_path) #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


if __name__ == "__main__":
    app.run(debug=True)
