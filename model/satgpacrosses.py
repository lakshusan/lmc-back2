import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

class SATtoGPAModel:
    _instance = None
    def __init__(self
    ):
        self.model = None
        self.dt = None
        self.features = ['time', 'kind']  # Removed 'id' and 'diet'
        self.target = 'pulse'
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.satscore_data = pd.DataFrame([])  # Initialize as an empty DataFrame
    @staticmethod
    def get_instance():   
        if SATtoGPAModel._instance is None:
            SATtoGPAModel._instance = SATtoGPAModel()
        return SATtoGPAModel._instance
    
    def init_satscore_list(self):
        self._init_satscore_data()  # Call _init_satscore_data first
        self._clean()  # Call data cleaning after initializing data
        self._train()  # Train the model after cleaning

    def _init_satscore_data(self):
        df = pd.read_csv("datasets/sattogpa.csv")
        self.satscore_data = pd.DataFrame(df)  # Convert df to DataFrame

    def _clean(self):
        if isinstance(self.satscore_data, pd.DataFrame):
            # Convert 'time' column to string type if it's not already
            self.satscore_data['time'] = self.satscore_data['time'].astype(str)
            # Split the 'time' values and take the first part, then convert it to integer
            self.satscore_data['time'] = self.satscore_data['time'].str.split().str[0].astype(int)
            # No need to drop 'id' and 'diet' columns here
            
            # Perform one-hot encoding on 'kind' column
            onehot = self.encoder.fit_transform(self.satscore_data[['kind']]).toarray()
            cols = ['kind_' + str(val) for val in self.encoder.categories_[0]]
            onehot_df = pd.DataFrame(onehot, columns=cols)
            
            # Check the DataFrame after one-hot encoding
            print("DataFrame after one-hot encoding:")
            print(onehot_df.head())
            
            # Check the categories learned by the encoder
            print("One-hot encoded categories:", self.encoder.categories_)
            
            # Concatenate the one-hot encoded columns with the existing DataFrame
            self.satscore_data = pd.concat([self.satscore_data, onehot_df], axis=1)
            self.features.extend(cols)  # Extend features list with one-hot encoded columns
            
            # Drop the original 'kind' column
            self.satscore_data.drop(['kind'], axis=1, inplace=True)
            # Remove 'kind' from features list
            self.features.remove('kind')
            
            self.satscore_data.dropna(inplace=True)
        else:
            raise ValueError("self.satscore_data is not a DataFrame.")
    
    def _train(self):
        # Training code remains the same
        X = self.satscore_data[self.features]
        y = self.satscore_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    def predict(self, person):
        if self.model is None:  # Check if model is trained
            self._train()  # Train model if not trained
        
        # Filter out features not present in the person dictionary
        features_to_keep = [feature for feature in self.features if feature in person]
        person_df = pd.DataFrame({feature: [person[feature]] for feature in features_to_keep})

        # Get the probability estimates for the positive class (index 1)
        positive_class_proba = self.model.predict_proba(person_df)[:, 1]
        return {'pulse': positive_class_proba}