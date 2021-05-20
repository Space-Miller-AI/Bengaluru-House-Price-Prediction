# Bengaluru House Price Prediction

## Table of Content
  * [Business Problem Statement](#Business-Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Cleaning](#Data-Cleaning)
  * [Model Building and Tuning](#Model-Building-and-Tuning)
   * [Other used Techniques](#Other-Used-Techniques)
  * [Demo](#demo)
  * [Motivation](#motivation)
  * [Installation](#installation)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Business Problem Statement
Buying a home, especially in a city like Bengaluru, is a tricky choice. While the major factors are usually the same for all metros, there are others to be considered for the Silicon Valley of India. With its help millennial crowd, vibrant culture, great climate and a slew of job opportunities, it is difficult to ascertain the price of a house in Bengaluru. As a result, this projects consists of predicting the house prices in Bengaluru, India using a dataset of more than 10000 records. Object oriented programming is used to build this project

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


## Other Used Techniques

## Demo
Link: [https://flightfareprediction2.herokuapp.com/](https://flightfareprediction2.herokuapp.com/)

[![](https://i.imgur.com/R1g2wvC.png)](https://flightfareprediction2.herokuapp.com/)

[![](https://i.imgur.com/p0aeL6c.png)](https://flightfareprediction2.herokuapp.com/)


## Motivation
What to do when you are at home due to this pandemic situation? I started to learn Machine Learning model to get most out of it. I came to know mathematics behind all supervised models. Finally it is important to work on application (real world application) to actually make a difference.

## Installation
The Code is written in Python 3.6.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.

[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── static 
│   ├── css
├── template
│   ├── home.html
├── Procfile
├── README.md
├── app.py
├── flight_price.ipynb
├── flight_rf.pkl
├── requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) 


## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Mandal-21/Flight-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope

* Use multiple Algorithms
* Optimize Flask app.py
* Front-End 
