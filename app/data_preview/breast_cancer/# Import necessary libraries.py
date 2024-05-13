# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import shap

# Load the dataset


data = pd.read_csv(r"./data_preview/breast_cancer/breast-cancer.txt")


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('diagnosis', axis=1),
                                                    data['diagnosis'], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Generate SHAP values for test data
explainer = shap.Explainer(model.predict_proba, X_train)
shap_values = explainer(X_test)

# Visualize SHAP values
shap.plots.waterfall(shap_values[0])

# Interpret the results
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
