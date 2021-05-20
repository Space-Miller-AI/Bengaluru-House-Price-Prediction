from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


class ModelTuner:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def train_model(self, model, X_train, y_train):
        try:
            self.logger_object.log(self.file_object, 'Entered the function train_model')
            model.fit(X_train, y_train)

            self.logger_object.log(self.file_object, 'Function train_model Completed Successfully! Exited this function.')
            return model
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function train_model. Error Messaeg : ' + str(e))


    def evaluate_model_regression(self, model, X, y):
        try:
            self.logger_object.log(self.file_object, 'Entered function evaluate_model_regression.')

            r2 = model.score(X, y)

            pred = model.predict(X)
            adj_r2 = r2_score(y, pred)

            mse = mean_squared_error(y, pred)
            mae = mean_absolute_error(y, pred)
            squareroot_mse = np.sqrt(mse)

            self.logger_object.log(self.file_object, f'R Squared : {r2}')
            self.logger_object.log(self.file_object, f'Adj R Squared : {adj_r2}')
            self.logger_object.log(self.file_object, f'MSE : {mse}')
            self.logger_object.log(self.file_object, f'MAE : {mae}')
            self.logger_object.log(self.file_object, f'Square Root MSE : {squareroot_mse}')

            self.logger_object.log(self.file_object, 'Function evaluate_model_regression COmpleted Successfully. Exited this function')
            return adj_r2

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function evaluate_model_regression. Error Message : ' + str(e))



    def evaluate_model_cross_validation(self, model, X_train, y_train, scoring, cv=5, verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered function evaluate_model_cross_validation.')
            scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, verbose=verbose)

            self.logger_object.log(self.file_object, 'Function evaluate_model_cross_validation Completed Successfully.Exited this function.')

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function evaluate_model_cross_validation. Error Message : ' + str(e))
        return scores



    def HyperparameterTuning_RandomizedSearchCV(self, X_train, y_train, model, params, scoring, n_iter=20, cv=5,
                                                verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered the function HyperparameterTuning_RandomizedSearchCV.')

            search = RandomizedSearchCV(model, params, n_iter=n_iter, scoring=scoring,
                                        cv=cv, n_jobs=-1, verbose=verbose, random_state=1)

            search.fit(X_train, y_train)
            model.set_params(**search.best_params_)

            dic = {'tuned_model': model,
                   'best_hyperparameters': search.best_params_,
                   'best_cv_score': search.best_score_}

            self.logger_object.log(self.file_object, 'Function HyperparameterTuning_RandomizedSearchCV Completed Successfully! Exited this function.')
            return dic

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function HyperparameterTuning_RandomizedSearchCV. Error message : ' + str(e))



    def best_single_model_RandomizedSearchCV(self, X_train, y_train, X_test, y_test, model_name, defaultModel, params,
                                             cv_scoring, cv_kfold=5, cv_n_iter=20):

        try:
            self.logger_object.log(self.file_object, 'Entered function best_single_model_RandomizedSearchCV. ')

            # create copies of default model object to avoid unwanted changes
            def_model = clone(defaultModel)
            tun_model = clone(defaultModel)

            # 1. Default Model
            # A) Evaluate model using cross validation
            default_model_cv_scores = self.evaluate_model_cross_validation(def_model, X_train, y_train, scoring=cv_scoring,
                                                                      cv=cv_kfold)
            default_model_cv_mean_score = np.mean(default_model_cv_scores)

            # B) Train KNN model with default hyperparameters
            default_model = self.train_model(def_model, X_train, y_train)

            # C) evaluate the model on test set and training set
            self.logger_object.log(self.file_object, f'Default {model_name} Model Performance on TEST SET : ')
            Path_DefaultModel_TestData = 'DefaultModel_TestData_ConfusionMatrix/'
            default_model_test_score = self.evaluate_model_regression(default_model, X_test, y_test)

            self.logger_object.log(self.file_object, f'Default {model_name} Model Performance on TRAINING SET : ')
            Path_DefaultModel_TrainingData = 'DefaultModel_TrainingData_ConfusionMatrix/'
            default_model_training_score = self.evaluate_model_regression(default_model, X_train, y_train)

            # 2. Tuned Model
            # If we want to use Optuna to tune Hyperparameters we call model_optuna function
            # A) Tune the model
            dic = self.HyperparameterTuning_RandomizedSearchCV(X_train=X_train, y_train=y_train, model=tun_model,
                                                          params=params, n_iter=cv_n_iter, scoring=cv_scoring,
                                                          cv=cv_kfold)
            tunedModel = dic['tuned_model']
            best_params = dic['best_hyperparameters']
            tuned_model_cv_mean_score = dic['best_cv_score']

            # B) Calculate the cross validation scores and mean on training set
            tuned_model_cv_scores = self.evaluate_model_cross_validation(tunedModel, X_train, y_train, scoring=cv_scoring,
                                                                    cv=cv_kfold)

            # B) Train the tuned model on training set
            tuned_model = self.train_model(tunedModel, X_train, y_train)

            # C) Evaluate the model on test set and training set
            self.logger_object.log(self.file_object, f'Tuned Model {model_name} Performance on TEST SET : ')
            Path_TunedModel_TestData = 'TunedModel_TestData_ConfusionMatrix/'
            tuned_model_test_score = self.evaluate_model_regression(tuned_model, X_test, y_test)

            self.logger_object.log(self.file_object, f'Tuned Model {model_name} Performance on TRAINING SET : ')
            Path_TunedModel_TrainingData = 'TunedModel_TrainingData_ConfusionMatrix/'
            tuned_model_training_score = self.evaluate_model_regression(tuned_model, X_train, y_train)

            # Find the best model with best score
            if tuned_model_test_score >= default_model_test_score:
                best_model_name = 'Tuned Model'
                best_model_object = tuned_model

            else:
                best_model_name = 'Default Model'
                best_model_object = default_model

            model_infos = {'Model Name': model_name,
                           'Default Model Object': default_model,
                           'Default Model Test Score': default_model_test_score,
                           'Default Model Training Score': default_model_training_score,
                           'Default Model CV Mean Score': default_model_cv_mean_score,
                           'Default Model CV Scores': default_model_cv_scores,
                           'Tuned Model Object': tuned_model,
                           'Best Hyperparameters': best_params,
                           'Tuned Model CV Mean Score': tuned_model_cv_mean_score,
                           'Tuned Model CV Scores': tuned_model_cv_scores,
                           'Tuned Model Test Score': tuned_model_test_score,
                           'Tuned Model Training Score': tuned_model_training_score,
                           'Final Best Model Name': best_model_name,
                           'Final Best Test Score': max(tuned_model_test_score, default_model_test_score),
                           'Final Best Model Object': best_model_object}

            self.logger_object.log(self.file_object, 'Function best_single_model_RandomizedSearchCV Completed Successfully. Exited this function.')
            return model_infos


        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function best_single_model_RandomizedSearchCV. Error message : ' + str(e))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression
    import random

    def best_model_OutOfManyModels_RandomizedSearchCV(self, X_train, y_train, X_test, y_test, cv_scoring, cv_kfold=5,
                                                      cv_niter=20, verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered the function best_model_OutOfManyModels_RandomizedSearchCV.')

            # model_df is dataframe which containss information about training and testing of each model
            model_df = pd.DataFrame()

            # dic is a dictionary which contains the best model object found for each ML algorithms and its score
            dic = {}

            # Find Best Linear Regression model. With Linear Regression there is no need to perform hyperparameter tuning
            print('Linear Regression Tuning')

            linear_regression_pipeline = Pipeline(
                [('feature_selector', SelectKBest(score_func=mutual_info_regression, k='all')),
                 ('feature_scaler', StandardScaler()),
                 ('estimator', LinearRegression())])

            linear_regression_params = {'feature_selector__k': [30, 35, 40, 43]}

            linear_regression_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                                           model_name='LinearRegression',
                                                                           defaultModel=linear_regression_pipeline,
                                                                           params=linear_regression_params,
                                                                           cv_scoring=cv_scoring, cv_kfold=cv_kfold,
                                                                           cv_n_iter=cv_niter)

            model_df = model_df.append(linear_regression_infos, ignore_index=True)

            # Get Best Linear Regression Model Object and its Test score
            best_linear_regression_object = linear_regression_infos['Final Best Model Object']
            best_linear_regression_test_score = linear_regression_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_linear_regression_object] = best_linear_regression_test_score

            # Find Best Random Forest Model
            randomforest_pipeline = Pipeline(
                [('feature_selector', SelectKBest(score_func=mutual_info_regression, k='all')),
                 ('feature_scaler', StandardScaler()),
                 ('estimator', RandomForestRegressor(random_state=1, n_estimators=300))])

            randomforest_params = {'feature_selector__k': [25, 30, 35, 40, 43],
                                   'estimator__max_depth': [3, 5, 7, 9, 12],
                                   'estimator__max_features': [None, 'sqrt', 'log2'],
                                   'estimator__min_samples_split': [2, 5, 10, 20, 30],
                                   'estimator__min_samples_leaf': [1, 5, 10, 20, 30],
                                   'estimator__criterion': ['mae', 'mse'],
                                   'estimator__bootstrap': [True, False]}

            print('Random Forest Tuning')
            randomforest_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                                      model_name='RandomForestRegressor',
                                                                      defaultModel=randomforest_pipeline,
                                                                      params=randomforest_params,
                                                                      cv_scoring=cv_scoring, cv_kfold=cv_kfold,
                                                                      cv_n_iter=cv_niter)

            # Add the RandomForest Infos to the dataframe
            model_df = model_df.append(randomforest_infos, ignore_index=True)
            # Get Best RandomForest Model Object and its test score
            best_randomforest_object = randomforest_infos['Final Best Model Object']
            best_randomforest_test_score = randomforest_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_randomforest_object] = best_randomforest_test_score

            # Find Best KNN Model
            knn_pipeline = Pipeline([('feature_selector', SelectKBest(score_func=mutual_info_regression, k='all')),
                                     ('feature_scaler', StandardScaler()),
                                     ('estimator', KNeighborsRegressor())])

            knn_params = {'feature_selector__k': [15, 25, 30, 40],
                          'estimator__n_neighbors': [1, 10, 20, 35, 45, 60],
                          'estimator__weights': ['uniform', 'distance']}

            print('KNN Tuning')
            knn_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                             model_name='KNeighborsRegressor',
                                                             defaultModel=knn_pipeline, params=knn_params,
                                                             cv_scoring=cv_scoring, cv_kfold=cv_kfold,
                                                             cv_n_iter=cv_niter)

            # Add the knn infos to the dataframe
            model_df = model_df.append(knn_infos, ignore_index=True)
            # Get Best KNN Model Object and its test score
            best_knn_object = knn_infos['Final Best Model Object']
            best_knn_test_score = knn_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_knn_object] = best_knn_test_score

            # save information about all models into a csv file and the information about xgboost in separate file if we have used early stopping
            model_df.to_csv('model_infos.csv', index=False)

            # Finding best model out of all models based on test score
            best_model_object = max(dic, key=dic.get)
            best_test_score = max(dic.values())

            try:
                self.logger_object.log(self.file_object,
                    'Function best_model_OutOfManyModels_RandomizedSearchCV Completed Successfully. Exited this function.')
                return best_model_object, str(best_model_object.named_steps['estimator']).split('(')[0], best_test_score

            except Exception as e:
                return best_model_object, str(best_model_object).split('(')[0], best_test_score

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function best_model_OutOfManyModels_RandomizedSearchCV. Error Message : ' + str(e))