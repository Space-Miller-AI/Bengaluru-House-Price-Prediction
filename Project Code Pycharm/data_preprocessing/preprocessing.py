from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import pylab
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Parameters : The dataframe and a list of column names to remove
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered the function remove_columns of the Preprocessor class')

        try:
            new_df = data.drop(columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Function remove_columns of class Preprocessor Completed Successfully. Exited this function.')
            return new_df
        except Exception as e:
            self.logger_object.log(self.file_object,'Error occured in function remove_columns of the Preprocessor class. Error message:  '+str(e))


    def separate_features_label(self, data, target_variable):
        """
            Method Name: separate_features_label
            Description: This method separates the features and labels from a dataset
            Parameters : entire dataset and target variable name
            Output: X dataframe with the features and y label
            On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered the function separate_features_label of class Preprocessor.')

        try:
            X = data.drop(target_variable, axis=1).copy()
            y = data[target_variable].copy()

            self.logger_object.log(self.file_object, 'Function separate_features_label of class Preprocessor Completed Succesfully. Exited this function.')
            return X, y

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function separate_features_label of class Preprocessor. Error Message : ' + str(e))


    def train_test_splitting(self, X, y, test_size, stratify=False):
        """
                    Method Name: train_test_splitting
                    Description: This method split the data into training and test data
                    Parameters : features dataframe, y label, size of test set, stratify=True if we want to obtain percetange of data in classification
                    Output: train and test dataframe
                    On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the function train_test_splitting of Preprocessor class.')

        try:
            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                    stratify=y, random_state=1)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                    random_state=1)

            #train_data = pd.concat([X_train, y_train], axis=1)
            #test_data = pd.concat([X_test, y_test], axis=1)

            self.logger_object.log(self.file_object, 'Function train_test_splitting of Preprocessor class Completed Successfully. Exited this function.')
            return X_train, X_test, y_train, y_test
            #return train_data, test_data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function train_test_splitting of Preprocessor class. Error Message : ' + str(e))





    def drop_rows_with_missing_values(self, data, subset_features, axis=0, how='any'):
        """
            Method Name: drop_missing_values
            Description: drops all rows that contain missing values in the specified features
            Parameters : dataset with features
            Output: new dataframe with dropped missing values
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the function drop_missing_values of Preprocessor class.')
        try:

            nr_rows_before = data.shape[0]
            new_data = data.dropna(axis=0, how='any', subset=subset_features)
            nr_rows_after = new_data.shape[0]

            self.logger_object.log(self.file_object, f'Nr of rows removed after dropping missing values : {nr_rows_before - nr_rows_after}')
            self.logger_object.log(self.file_object, 'Function drop_missing_values of Preprocessor class Completed Succesfully. Exited this function.')
            return new_data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function drop_missing_values of Preprocessor class. Error message : ' + str(e))


    def fill_nan_missing_string(self, data, features):
        """
            Method Name: fill_nan_missing_string
            Description: fills nan values with string 'Missing' in specified features
            Parameters : dataset with features
            Output: new dataframe with imputed nan values
            On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered function fill_nan_missing_string of Preprocessor class.')
        try:
            for feature in features:
                data[feature] = data[feature].fillna('Missing')


            self.logger_object.log(self.file_object, 'Function fill_nan_missing_string of Preprocessor class Completed Successfully. Exited this function.')
            return data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function fill_nan_missing_string of Preprocessor class. Error MEssage : ' + str(e))

    def check_for_missing_values(self, dataset):
        try:

            self.logger_object.log(self.file_object, 'Entered function check_for_missing_values of Preprocessor class.')
            if dataset.isna().sum().any():

                df_nan_values = pd.DataFrame()

                for col in dataset.columns:
                    nr_nan_values_percentage = dataset[col].isna().mean()
                    nr_nan_values = dataset[col].isna().sum()

                    if nr_nan_values > 0:
                        row = {'Feature': col,
                               'Percentage Nan Values': nr_nan_values_percentage,
                               'Amount Nan Values': nr_nan_values}
                        df_nan_values = df_nan_values.append(row, ignore_index=True)

                df_nan_values.to_csv('nan_values.csv', index=False)

                self.logger_object.log(self.file_object,
                                       'Function check_for_missing_values of Preprocessor class Completed Successfully. Exited this function.')
                return True

            else:
                self.logger_object.log(self.file_object, 'Function check_for_missing_values of Preprocessor class Completed Successfully. Exited this function.')
                return False


        except Exception as e:
            self.logger_object.log(self.file_object, 'Error Occured in function check_for_missing_values of Preprocessor class. Error Message : ' + str(e))




    def gaussian_transformation(self, X_train, X_test):
        try:
            self.logger_object.log(self.file_object, 'Entered function gaussian_transformation of Preprocessor class.')

            X_train['total_sqft'] = np.log(X_train['total_sqft'])
            X_train['bath'], parameters = stat.boxcox(X_train['bath'])

            X_test['total_sqft'] = np.log(X_test['total_sqft'])
            X_test['bath'], parameters = stat.boxcox(X_test['bath'])

            self.logger_object.log(self.file_object, 'Function gaussian_transformation  of Preprocessor class. Completed Successfully. Exited this function.')
            return X_train, X_test

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function gaussian_transformation of Preprocessor class. Error message : ' + str(e))


    def encode_categorical_features(self, data):
        """
            Method Name: encode_categorical_features
            Description: encoding categorical features into numerical
            Parameters : dataset with features
            Output: new dataframe with only numerical features
            On Failure: Raise Exception
        """
        try:
            self.logger_object.log(self.file_object, 'Entered function encode_categorical_features of Preprocessor class.')

            # encoding society
            data = self.reduce_categories(data, 'society', 20)
            data = pd.get_dummies(data, columns=['society'], drop_first=True)

            # encoding area_type
            data = pd.get_dummies(data, columns=['area_type'], drop_first=True)

            # encoding size feature
            data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]))
            data = self.remove_columns(data, ['size'])

            # encoding total_sqft feature
            data['total_sqft'] = data['total_sqft'].apply(self.convert_sqft_to_num)

            # encoding location feature
            data = self.reduce_categories(data, 'location', 20)
            location_onehot = pd.get_dummies(data['location'], drop_first=True)
            data = pd.concat([data, location_onehot], axis=1)

            self.logger_object.log(self.file_object,
                               'Function encode_categorical_features of Preprocessor class Completed Successfully. Exited this function.')

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,
                               'Error occured in function encode_categorical_features of Preprocessor class. Error Message : ' + str(
                                   e))

    def reduce_categories(self, dataset, feature, top_k):


        try:
            self.logger_object.log(self.file_object, 'Entered function reduce_categories of Preprocessor class.')

            top_k_frequencies = dataset[feature].value_counts(ascending=False)[:top_k]
            top_k_categories = top_k_frequencies.index
            total_nr_categories = len(dataset[feature].unique())
            nr_categories_other = len([cat for cat in dataset[feature].unique() if cat not in top_k_categories])
            self.logger_object.log(self.file_object,
                f"Nr of categories (low frequency categories) labeled with 'other' : {nr_categories_other} out of total {total_nr_categories}")
            self.logger_object.log(self.file_object, f"Nr of categories not labeled with 'other' : {top_k}")
            dataset[feature] = dataset[feature].apply(lambda x: x if x in top_k_categories else 'other')

            #dataset = pd.get_dummies(dataset, columns=[feature], drop_first=True)

            self.logger_object.log(self.file_object, 'Function reduce_categories of Preprocessor class Completed Successfully. Exited this function.')
            return dataset
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function reduuce_categories of Preprocessor class. Error Message : '+str(e))


    def convert_sqft_to_num(self, x):
        try:

            try:
                float_x = float(x)
                return float_x

            except:

                tokens = x.split('-')
                if len(tokens) == 2:
                    return (float(tokens[0]) + float(
                        tokens[1])) / 2

                tokens = x.split(' ')
                if len(tokens) == 2:
                    if tokens[1] == 'Yards':
                        yards = float(tokens[0][:-3])
                        sqft = yards / 0.11111111
                        return sqft

                    elif tokens[1] == 'Meter':
                        meters = float(tokens[0][:-3])
                        sqft = meters / 0.09290304
                        return sqft

                else:
                    if 'Acres' in x:
                        tokens = x.split('Acres')
                        acres = float(tokens[0])
                        sqft = acres / 0.00002296
                        return sqft

                    elif 'Guntha' in x:
                        tokens = x.split('Guntha')
                        guntha = float(tokens[0])
                        sqft = guntha / 0.00000003587
                        return sqft

                    elif 'Cents' in x:
                        tokens = x.split('Cents')
                        cents = float(tokens[0])
                        sqft = cents / 0.0023
                        return sqft

                    elif 'Grounds' in x:
                        tokens = x.split('Grounds')
                        grounds = float(tokens[0])
                        sqft = grounds / 0.00041666666666667
                        return sqft

                return None

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function convert_sqft_to_num of Preprocessor class. Error Message : ' + str(e))



    def handling_outliers(self, data):

        try:
            self.logger_object.log(self.file_object, 'Entered function handling_outliers of Preprocessor class.')

            # add new feature price_per_sqft that we need to handle outliers
            data['price_per_sqft'] = data['price']*100000 / data['total_sqft']

            # A) Outlier Removal : remove all houses where sqft per 1 bedroom is less than 300
            nr_rows1 = data.shape[0]
            data = data[~(data.total_sqft / data.bhk < 300)]
            nr_rows2 = data.shape[0]
            self.logger_object.log(self.file_object, f'Nr of rows dropped after removing all houses where sqft per 1 bedroom is less than 300 : {nr_rows1-nr_rows2}')

            # B) Outlier Removal : remove all houses per location where price per sqft is greater
            # than mean + 1std or less than mean - 1std
            data = self.remove_pps_outliers(data)
            nr_rows3 = data.shape[0]
            self.logger_object.log(self.file_object, f'Nr of rows dropped after removing all houses per location where price per sqft is greater than mean + 1std or less than mean - 1std : {nr_rows2-nr_rows3}')


            # C) Outlier Removal : remove all houses where price total price of x+1 BHK is greater than prices of houses with x BHK for same total sqft.
            data = self.remove_bhk_outliers(data)
            nr_rows4 = data.shape[0]
            self.logger_object.log(self.file_object, f'Nr of rows dropped after removing all houses where price total price of x+1 BHK is greater than prices of houses with x BHK for same total sqft: {nr_rows3-nr_rows4}')

            # D) Outlier Removal : remove all houses where nr of bathrooms is greater than 2 + nr of bedroom
            data = data[data.bath < data.bhk + 2]
            nr_rows5 = data.shape[0]
            self.logger_object.log(self.file_object, f'Nr of rows dropped after removing all houses where nr of bathrooms is greater than 2 + nr of bedroom : {nr_rows4-nr_rows5}')


            # remove columns price_per_sqft and location that we needed only for outlier treatment
            data = self.remove_columns(data, ['price_per_sqft', 'location'])

            self.logger_object.log(self.file_object, 'Function remove_outliers of Preprocessor class Completed Successfully. Exited this function.')
            return data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function handling_outliers of Preprocessor class. Error Message : ' + str(e))

    def remove_pps_outliers(self, df):

        try:
            self.logger_object.log(self.file_object, 'Entered function remove_pps_outliers of Preprocessor class.')
            df_out = pd.DataFrame()
            for key, subdf in df.groupby('location'):
                m = np.mean(subdf.price_per_sqft)
                st = np.std(subdf.price_per_sqft)
                reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]

                df_out = pd.concat([df_out, reduced_df], ignore_index=True)

            self.logger_object.log(self.file_object, 'Function remove_pps_outliers of Preprocessor class Completed Successfully. Exited this function.')
            return df_out

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function remove_pps_outliers of Preprocessor class. Error Message : ' + str(e))

    def remove_bhk_outliers(self, df):

        try:
            self.logger_object.log(self.file_object, 'Entered function remove_bhk_outliers of Preprocessor class. ')
            exclude_indices = np.array([])
            for location, location_df in df.groupby('location'):
                bhk_stats = {}
                for bhk, bhk_df in location_df.groupby('bhk'):
                    bhk_stats[bhk] = {
                        'mean': np.mean(bhk_df.price_per_sqft),
                        'std': np.std(bhk_df.price_per_sqft),
                        'count': bhk_df.shape[0]
                    }
                for bhk, bhk_df in location_df.groupby('bhk'):
                    stats = bhk_stats.get(bhk - 1)
                    if stats and stats['count'] > 5:
                        exclude_indices = np.append(exclude_indices,
                                                    bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)

            self.logger_object.log(self.file_object, 'Function remove_bhk_outliers of Preprocessor class Completed Successfully. Exited this function.')
            return df.drop(exclude_indices, axis='index')

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function remove_bhk_outliers of Preprocessor class. Error message : ' + str(e))


    def dropping_constant_features(self, X_train, X_test, threshold):

        try:
            self.logger_object.log(self.file_object, 'Entered function dropping_constant_features of Preprocessor class. ')
            var_thres = VarianceThreshold(threshold=threshold)
            var_thres.fit(X_train)

            cols = X_train.columns[var_thres.get_support()]
            features_low_variance = [feature for feature in X_train.columns if feature not in cols]
            self.logger_object.log(self.file_object,
                f'Features with variance={threshold} : {features_low_variance}. Amount : {len(features_low_variance)} out of {len(X_train.columns)} total features')

            X_train = X_train[cols]
            X_test = X_test[cols]

            self.logger_object.log(self.file_object, 'Function dropping_constant_features of Preprocessor class Completed Successfully. Exited this functiin.')
            return X_train, X_test
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function dropping_constant_features of Preprocessor class. Error MEssage : ' + str(e))



    def correlation_heatmap(self, X_train):
        self.logger_object.log(self.file_object, 'Entered function multicolleniarity of Preprocessor class.')
        try:
            plt.figure(figsize=(30, 22))
            cor = X_train.corr()
            sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
            plt.savefig('multicolleniarity_heatmap.jpg')

            self.logger_object.log(self.file_object, 'Function multicolleniarity Completed Successfully. Exited this function.')
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function multicolleniarity of Preprocessor class. Error Message : ' + str(e))

    def detect_multicolleniarity(self, dataset, threshold):
        try:
            self.logger_object.log(self.file_object, 'Entered function detect_multicolleniarity of Preprocessor class. ')
            pairs = set()
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        colname1 = corr_matrix.columns[i]
                        colname2 = corr_matrix.columns[j]
                        pairs.add((colname1, colname2, corr_matrix.iloc[i, j]))


            self.logger_object.log(self.file_object, 'High correlated indepedent features are : ' + str(pairs))
            self.logger_object.log(self.file_object, 'Functiondetect_multicolleniarity of Preproccessor class Completed Successfully. Exited this function.')
            return pairs

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function detect_multicolleniarity of Preprocessor class. Error Message : ' + str(e))


    def remove_correlated_independent_features(self, X_train, y_train, X_test, features_correlated, target_feature):
        try:
            self.logger_object.log(self.file_object, 'Enterd function remove_correlated_independent_features of Preprocessor class.')
            train_data = pd.concat([X_train, y_train], axis=1)
            corr_with_target = abs(train_data.corr()[target_feature][features_correlated])
            logical_vector = abs(train_data.corr()[target_feature]) == np.max(corr_with_target)
            feature_highest_correlation_with_target = train_data.corr()[target_feature].index[logical_vector][0]
            features_to_remove = [feature for feature in features_correlated if
                                  feature not in feature_highest_correlation_with_target]


            self.logger_object.log(self.file_object, f'Feature with Highest Correlation with Target : {feature_highest_correlation_with_target}')
            self.logger_object.log(self.file_object, f'Features to remove : {features_to_remove}')

            new_X_train = X_train.drop(columns=features_to_remove, axis=1)
            new_X_test = X_test.drop(columns=features_to_remove, axis=1)

            self.logger_object.log(self.file_object, 'Function remove_correlated_independent_features of Preprocessor class Completed Successfully. Exited this function')
            return new_X_train, new_X_test

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function remove_correlated_independent_features of Preprocessor class. Error Message : ' + str(e))




    def feature_scaling(self, X_train, X_test, cols_to_scale, one_hot_cols):
        try:
            print('Entered function feature_scaling of Preprocessor class.')

            # scaling training features
            X_train_cols_to_scale = X_train[cols_to_scale]
            X_train_one_hot_cols = X_train[one_hot_cols]

            cols = X_train_cols_to_scale.columns
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_cols_to_scale)
            X_train = pd.DataFrame(data=X_train_scaled, columns=cols)
            new_X_train = pd.concat([X_train, X_train_one_hot_cols.reset_index()], axis=1)

            # scaling test features
            X_test_cols_to_scale = X_test[cols_to_scale]
            X_test_one_hot_cols = X_test[one_hot_cols]

            cols = X_test_cols_to_scale.columns
            X_test_scaled = scaler.transform(X_test_cols_to_scale)
            X_test = pd.DataFrame(data=X_test_scaled, columns=cols)
            new_X_test = pd.concat([X_test, X_test_one_hot_cols.reset_index()], axis=1)

            print('Function feature_scaling of Preprocessor class Completed Succesfully. Exited this function.')
            return new_X_train, new_X_test

        except Exception as e:
            print('Error occured in function feature_scaling of Preprocessor class. Error Message:' + str(e))