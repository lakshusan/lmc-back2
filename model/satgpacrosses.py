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
        self.features = ['SAT Score', 'Grade Point Average']
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
        df = pd.read_csv("sattogpa.csv")
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

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

''' Tutorial: https://www.sqlalchemy.org/library.html#tutorials, try to get into Python shell and follow along '''

df = pd.read_csv('sattogpa.csv')
dataList = ['SAT Score' 'Grade Point Average']
labelencoder = LabelEncoder()
X = df.drop(columns=['SAT Score'])
y = df['Grade Point Average']
regressor = RandomForestRegressor(n_estimators=10, random_state=42)
regressor.fit(X, y)

def stringToInt(var):
    if var == 'yes':
        var = 1
    elif var == 'no':
        var = 0
    else:
        var = int(var)
    return var

class SATtoGPAModel():
    def __init__(self, satscore, GPA):
        self._satscore = satscore
        self._GPA = GPA

    @property
    def satscore(self):
        return self._satscore

    @satscore.setter
    def satscore(self, value):
        self._satscore = value

    @property
    def GPA(self):
        return self._GPA

    @GPA.setter
    def GPA(self, value):
        self._GPA = value

    # @property
    # def play_badminton(self):
    #     return self._play_badminton

    # @play_badminton.setter
    # def play_badminton(self, value):
    #     self._play_badminton = value

    def predict(self):
        # Ensure all features are properly converted
        self._satscore = stringToInt(self._satscore)
        self._GPA = stringToInt(self._GPA)

        varList = [self._satscore, self._GPA]

        # Ensure the same number of features as the model expects

        # Predicting the price
        predicted_badminton = regressor.predict([varList])[0]
        if predicted_badminton == 0:
            string = 'no'
        else:
            string = 'yes'
        return string