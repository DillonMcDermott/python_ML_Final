import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("car_price_prediction_.csv")

# Quick look at data
print(df.head())
print(df.info())
print(df.describe())

# Check column names to ensure "Brand" column exists and no extra spaces
print("Columns before any processing:", df.columns)

# Strip any leading/trailing spaces from the column names
df.columns = df.columns.str.strip()

# Check again after stripping
print("Columns after stripping spaces:", df.columns)

# Visualize distribution of the Price
plt.figure(figsize=(8, 4))
sns.histplot(df["Price"], bins=50, kde=True)
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Feature Engineering

df = df.drop(["Car ID", "Model"], axis=1)

# Add car age instead of using Year directly
df["Car_Age"] = 2025 - df["Year"]
df = df.drop("Year", axis=1)

# Change Columns into numbers
df = pd.get_dummies(df, columns=["Brand", "Transmission", "Fuel Type", "Condition"], drop_first=True)

# Remove outliers in price
df = df[(df["Price"] >= 7000) & (df["Price"] <= 90000)]

# Drop rows with any missing values
df = df.dropna()

# Check if "Brand" column exists after transformation
print("Columns after feature engineering:", df.columns)

# Split into features (X) and target (y)
X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Train and Evaluate Models =====

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42
    ),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),
    "Linear Regression": LinearRegression()
}

results = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    
    # Predicting the prices
    y_pred = np.expm1(model.predict(X_test))

    y_test_actual = np.expm1(y_test)

    # Calculate MAE (mean_absolute_error) and R^2
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    print(f"{name} - Mean Absolute Error: {mae:.2f}")
    print(f"{name} - R² Score: {r2:.2f}")

    results[name] = {"MAE": mae, "R2": r2}

    # Plot results
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_actual, y_pred, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"{name} — Actual vs Predicted Car Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Simple baseline: predict mean price
baseline_pred = [np.expm1(y_train).mean()] * len(y_test)  # use expm1 to reverse the log1p transform
baseline_mae = mean_absolute_error(np.expm1(y_test), baseline_pred)

print(f"\nBaseline Mean Absolute Error (predicting mean price): {baseline_mae:.2f}")


# Rank Models
best_mae_model = min(results, key=lambda x: results[x]["MAE"])
best_r2_model = max(results, key=lambda x: results[x]["R2"])

print(f"Best Model Based on MAE: {best_mae_model} (MAE: {results[best_mae_model]['MAE']:.2f})")
print(f"Best Model Based on R²:  {best_r2_model} (R²: {results[best_r2_model]['R2']:.2f})")
