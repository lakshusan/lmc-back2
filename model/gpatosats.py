import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class GPAtoSATModel:
    def __init__(self, gpa):
        self._gpa = gpa
        # self.target = 'satscore'

    def predict(self):
        df = pd.read_csv('gpatosat.csv')
        X = df['Grade Point Average'] # only input
        y = df['SAT Score']
        regressor = RandomForestRegressor(n_estimators=10, random_state=42)
        regressor.fit(X.values.reshape(-1, 1), y)  # Reshape X to be a 2D array
        # Predicting the GPA
        predicted_satscore = regressor.predict([[self._gpa]])[0]
        return predicted_satscore