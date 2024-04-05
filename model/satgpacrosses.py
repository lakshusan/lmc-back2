import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load the fitness dataset into a pandas DataFrame
data = pd.read_csv("/home/lakshusan/vscode/lmc-back2/datasets/sattogpa.csv")

train_data, test_data = train_test_split(data, test_size=5, random_state=42)

# Print training data
print("Training Data:")
print(train_data)

# Print test data
print("\nTest Data:")
print(test_data)

X_train = train_data[['SAT Score']]
y_train = train_data['Grade Point Average']
X_test = test_data[['SAT Score']]
y_test = test_data['Grade Point Average']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

def calculate_estimate():
    SATscore = float(input("Enter SAT Score (out of 1600): "))

    # Make a prediction using the user input
    prediction = model.predict([[SATscore]])

    print("SAT Score:",SATscore)
    print("Grade Point Average Estimate:", prediction[0])

    actual_GPA = float(input("Enter Actual Grade Point Average: "))
    print("Actual Grade Point Average:", actual_GPA)
    percent_accuracy = 100 * (1 - abs(prediction[0] - actual_GPA) / actual_GPA)
    print("Percent Accuracy: {:.2f}%".format(percent_accuracy))

# Prompt user for input and calculate estimate
calculate_estimate()