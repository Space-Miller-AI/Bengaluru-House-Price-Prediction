# Bengaluru House Price Prediction

## Table of Content
  * [Business Problem Statement](#Business-Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Cleaning](#Data-Cleaning)
  * [Model Building and Tuning](#Model-Building-and-Tuning)
  * [Other used Techniques](#Other-Used-Techniques)
  * [Demo](#demo)
  * [Run project in your local machine](#Run-the-project-in-your-local-machine)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Business Problem Statement
Buying a home, especially in a city like Bengaluru, is a tricky choice. While the major factors are usually the same for all metros, there are others to be considered for the Silicon Valley of India. With its help millennial crowd, vibrant culture, great climate and a slew of job opportunities, it is difficult to ascertain the price of a house in Bengaluru. As a result, this projects consists of predicting the house prices in Bengaluru, India using a dataset of more than 10000 records.
## Data
Data Source : Kaggle.

Link : https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data

## Used Libraries and Resources
**Python Version** : 3.6

**Libraries** : sklearn, pandas, numpy, matplotlib, seaborn, flask, json, pickle

**References** : https://towardsdatascience.com/, https://machinelearningmastery.com/


## Data Cleaning
The interesting thing of this project is that the data is really dirty and has many outliers. So it needs to do a lot of feature engineering to prepare for a machine learning model. I made the following changes :

* Based on domain knowledge I removed columns which are not relevant for prediction.
* Removed the missing values in columns where the percentage of missing values was very small while in other columns I labeled them with the string 'Missing'.
* Categorical features with only few different categories are encoded using one-hot encoding technique while for features with huge amount of categories are considered only the top 20 most frequent categories. The other ones are labeled with 'other' are then one-hot encoding is applied to encode them into numerical features.
* Some features like number of bedrooms have object datatype so I converted them into float numbers.
* Total_sqft feature which shows the total square foot of a house is measured in different units like Sq. Yards, Sq. Meter, Acres, Guntha, Cents and have different datatypes like float, integer, string etc. All this values are converted into float number and the same unit which is square foot.
* Outlier Removal 1: Business manager (who has expertise in real estate), told me that the minimal square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. So based on business logic I removed all houses where sqft per 1 bedroom is less then 300.
* Outlier Removal 2: Based on business logic I removed all houses where price per sqft per location is greater than mean + 1 std and less than mean - 1 std.
* Outlier Removal 3: Based on business logic I removed all houses where total price of x+1 BHK is greater than prices of houses with x BHK for same total sqft.
* Outlier Removal 4: Based on business logic I removed all houses where nr of bathrooms is greater than 2 + nr of bedroom.
* Since one of the ML estimators I trained is Linear Regression, I applied some Gaussian transformation techniques on some features to convert them into normally distributed features.
* Features with variance=0 (constant features) are dropped since they do not give any useful information about the target variable.
* Since one of the ML estimators I trained is Linear Regression I handled multicolleniarity by dropping independent features (redundant features) that are highly correlated with each other.
* Feature Selection and Feature Scaling is performed together with the hyperparameter tuning using sklearn pipelines in order to avoid data leakage.


## Model Building and Tuning

* The ML Estimators I trained are : Linear Regression, Random Forest and KNN.
* Evaluated the models using R Squared, Adjusted R Squared, Mean Absolute Error, Mean Squared Error, Root Mean Squared Error. The performance metric used to select the best model is Adjusted R Squared.
* When tuning the model using Cross Validation, I used sklearn pipeline including feature selection, feature scaling and hyperparameter tuning in order to avoid data leakage. So I tuned the nr of features and different hyperparameters in each fold of cross validation.
* Hyperparameter Tuning  is done using RandomizedSearchCV.
* I evaluated each ML model using training score, cross validation mean score, cross validation scores, test score to get a better understanding about the model performances. The best model is selected using the test score.
* The best model I got from optimization is Random Forest with a test score of 0.879
* Every information about different performance metrics of default (model with default hyperparameters) and tuned models training is stored in Training Infos.csv file.

| Model Name    | Test Score    | Training Score  | 
| ------------- |:-------------:|:---------------:|
| col 3 is      | right-aligned | $1600           |
| col 2 is      | centered      |   $12           |
| zebra stripes | are neat      |    $1           |

## Other Used Techniques

* Object oriented programming is used to build this project in order to create modular and flexible code.
* A retraining approach is implemented using Flask framework.
* Using Logging every information about data cleaning und model training HISTORY (since we may train the model many times using retraining approach)  is stored is some txt files and csv files for example : the amount of missing values for each feature, the amount of records removed after dropping the missing values and outliers, the amount of at least frequent categories labeled with 'other' during encoding, the dropped constant features, highly correlated independent features, which features are dropping during handling multicolleniarity, best selected features, model accuracies and errors etc.

## Demo

This is how the web application looks like : 


![alt text](https://github.com/Lori10/Banglore-House-Price-Prediction/blob/master/Project%20Code%20Pycharm/demo_image.jpg "Image")



## Run the project in your local machine 

1. Clone the repository
2. Open the project directory (PyCharm Project Code foler) in PyCharm  and create a Python Interpreter using Conda Environment : Settings - Project : Project Code Pycharm - Python Interpreter - Add - Conda Environment - Select Python Version 3.6 - 
3. Run the following command in the terminal to install the required packages and libraries : pip install -r requirements.txt
4. Run the file app.py by clicking Run and open the API that shows up in the bottom of terminal.


## Directory Tree 
```
 ├── Project Code PyCharm├── static 
                             ├── css
                                 ├── styles.css
                         ├── templates
                         │   ├── home.html
                         ├── File_Operation
                             ├── FileOperation.py
                         ├── Functions
                             ├── functions.py
                         ├── Logs
                             ├── DataPreprocessing_Log.txt
                             ├── ModelTraining_Log.txt
                             ├── Prediction_Log.txt
                         ├── ModelTraining
                             ├── trainingModel.py
                         ├── Training_FileFromDB
                             ├── dataset.csv
                         ├── application_logging
                             ├── logger.py
                         ├── best_model_finder
                             ├── modelTuning.py
                         ├── data_ingestion
                             ├── data_loader.py
                         ├── data_preprocessing
                             ├── preprocessing.py
                         ├── models
                             ├── RandomForestRegressor
                                 ├── RandomForestRegressor.sav
                         ├── app.py
                         ├── encoded_features.json
                         ├── model_infos.csv
                         ├── multicolleniarity_heatmap.jpg
                         ├── nan_values.csv
                         ├── Training Infos.ipynb
                         ├── requirements.txt
```



## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Lori10/Banglore-House-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope

* Use other ML Estimators
* Try other feature engineering approaches to get a possible higher model performance
* Optimize Flask app.py Front End
