from application_logging.logger import App_Logger
from data_ingestion.data_loader import Data_Getter
from data_preprocessing.preprocessing import Preprocessor
from best_model_finder.modelTuning import  ModelTuner
from xgboost import XGBClassifier
from File_Operation.FileOperation import File_Operations
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action='ignore')

class trainModel:

    def __init__(self, training_file_path):
        self.logger = App_Logger()
        self.file_training = open("Logs/ModelTraining_Log.txt", 'a+')
        self.file_preprocessing = open('Logs/DataPreprocessing_Log.txt', 'a+')
        self.training_file_path = training_file_path

    def trainingModel(self):
        # Logging the start of Training
        self.logger.log(self.file_preprocessing, 'Start Preprocessing and Training')
        try:

            # DATA PREPROCESSING

            # read the csv file (dataset)
            dg = Data_Getter(self.file_preprocessing, self.logger, self.training_file_path)
            df = dg.get_data()

            # FEATURE ENGINEERING
            preprocessor = Preprocessor(file_object=self.file_preprocessing, logger_object=self.logger)

            # 1. Dropping columns that are not relevant for prediction (we apply for training and test data)
            df = preprocessor.remove_columns(df, ['availability'])

            # 2. Handling Missing Values
            # 2.1 Check if there are any missing values
            result = preprocessor.check_for_missing_values(df)
            if result:
                # 2.2 Dropping rows with missing values in those specified columns
                df = preprocessor.drop_rows_with_missing_values(data=df, axis=0, how='any',
                                                                subset_features=['location', 'size', 'bath', 'balcony'])

                # 2.3 Filling missing values with 'Missing' in column 'society'
                df = preprocessor.fill_nan_missing_string(df, ['society'])

            # 3. Encoding categorical features
            df = preprocessor.encode_categorical_features(df)

            # 4. Handling Outliers
            df = preprocessor.handling_outliers(df)

            # 5. Train Test Split
            X, y = preprocessor.separate_features_label(df, 'price')
            X_train, X_test, y_train, y_test = preprocessor.train_test_splitting(X, y, test_size=0.2)

            # 6. Gaussioan (Normal Distribution) Transformation
            X_train, X_test = preprocessor.gaussian_transformation(X_train, X_test)

            # FEATURE SELECTION
            # 6. Dropping constant features
            X_train, X_test = preprocessor.dropping_constant_features(X_train, X_test, threshold=0)

            # 7. Handling Multicolleniarity
            # 7.1 Detecting Multicolleniarity
            preprocessor.correlation_heatmap(X_train)
            corr_features = preprocessor.detect_multicolleniarity(X_train, 0.71)

            # 7.2 Handling Multicolleniarity : removing redudant independent features
            X_train, X_test = preprocessor.remove_correlated_independent_features(X_train, y_train, X_test,
                                                                                  ['bath', 'total_sqft', 'bhk'],
                                                                                  'price')
            X_train, X_test = preprocessor.remove_correlated_independent_features(X_train, y_train, X_test,
                                                                                  ['society_other', 'society_Missing'],
                                                                                  'price')

            # 8. Model Tuning
            model_tuner = ModelTuner(self.file_training, self.logger)
            best_model_object, best_model_name, best_test_score = model_tuner.best_model_OutOfManyModels_RandomizedSearchCV(
                X_train, y_train, X_test, y_test,
                cv_scoring='r2', cv_niter=30)

            self.logger.log(self.file_training, f'Best Model Name : {best_model_name}. Best Test Score : {best_test_score}')

            # 9. Save Model to a pickle file
            op = File_Operations(self.file_training, self.logger)
            op.save_model(best_model_object, best_model_name)

            self.logger.log(self.file_training, 'Successful End of Training!')
            self.file_training.close()
            self.file_preprocessing.close()

        except Exception as e:
            self.logger.log(self.file_training, 'Unsuccessful End of Training. Error Message : ' + str(e))
            self.file_training.close()
            self.file_preprocessing.close()