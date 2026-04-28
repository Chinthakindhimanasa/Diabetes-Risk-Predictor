import pandas as pd
import joblib
import shap

# Load dataset
df = pd.read_csv("cleaned_diabetes.csv")

X = df.drop("Outcome", axis=1)

# Load model
model = joblib.load("model.pkl")

# SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Show graph
shap.summary_plot(shap_values, X)