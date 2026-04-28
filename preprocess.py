import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv")

# Check missing values
print("Missing values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)

# Save cleaned dataset
df.to_csv("cleaned_diabetes.csv", index=False)

print("\nPreprocessing completed successfully")