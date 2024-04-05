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
        self.features = ['SAT Score', 'Grade Point Average']  # Removed 'id' and 'diet'
        self.target = 'Grade Point Average'
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
            # Check if 'Grade Point Average' column exists
            if 'Grade Point Average' in self.satscore_data.columns:
                # Convert 'Grade Point Average' column to float type
                self.satscore_data['Grade Point Average'] = self.satscore_data['Grade Point Average'].astype(float)
                # Drop any rows with missing values in the 'Grade Point Average' column
                self.satscore_data.dropna(subset=['Grade Point Average'], inplace=True)
                # Optionally, round the GPA values if you need them as integers
                # self.satscore_data['Grade Point Average'] = self.satscore_data['Grade Point Average'].round().astype(int)
            else:
                raise ValueError("Column 'Grade Point Average' not found in the dataset.")
        else:
            raise ValueError("self.satscore_data is not a DataFrame.")
    
    def _train(self):
        # Training code using Linear Regression
        X = self.satscore_data[self.features]
        y = self.satscore_data[self.target]
        # Initialize a Linear Regression model
        self.model = LinearRegression()
        self.model.fit(X, y)


    def predict(self, person):
        if self.model is None:  # Check if model is trained
            self._train()  # Train model if not trained
    
        # Filter out features not present in the person dictionary
        features_to_keep = [feature for feature in self.features if feature in person]
        person_df = pd.DataFrame({feature: [person[feature]] for feature in features_to_keep})

        # Get the probability estimates for the positive class (index 1)
        gpa_estimate = self.model.predict(person_df)[0]  # Assuming the GPA is directly predicted
        return {'GPA_estimate': gpa_estimate}
