import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'marks': [20, 30, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

# Features and target
X = df[['hours']]
y = df['marks']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predicted_marks = model.predict([[5]])
print("Predicted Marks:", predicted_marks)
