import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
file_path = "house_price_pred_train.csv"  # Ensure the correct path
data = pd.read_csv(file_path)

# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Calculate correlation with the target variable (SalePrice)
correlation_matrix = numeric_data.corr()
correlation_with_target = correlation_matrix["SalePrice"].sort_values(ascending=False)

# Select the top 8 features most correlated with SalePrice (excluding SalePrice itself)
top_features = correlation_with_target.index[1:9].tolist()  # Exclude 'SalePrice'

print("Selected Top Features:")
print(top_features)

# Ensure there are no missing values for selected features
selected_data = data[top_features + ["SalePrice"]].dropna()

# Split the data into features (X) and target (y)
X = selected_data[top_features]
y = selected_data["SalePrice"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a pickle file
model_path = "model/house_price_model.pkl"  # Ensure the `model` directory exists
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model trained and saved at {model_path}")


############################## Calculate min and max for each feature #################################3
# Convert feature ranges to JSON-serializable format
feature_ranges = {
    feature: {
        "min": float(selected_data[feature].min()),
        "max": float(selected_data[feature].max())
    }
    for feature in top_features
}

# Save feature ranges as a JSON file
import json
ranges_path = "model/feature_ranges.json"
with open(ranges_path, "w") as file:
    json.dump(feature_ranges, file)

print("Feature ranges saved for use in the form.")
