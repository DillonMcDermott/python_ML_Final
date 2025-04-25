import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("car_price_prediction_.csv")

# Feature Engineering
df = df.drop(["Car ID", "Model"], axis=1)
df["Car_Age"] = 2025 - df["Year"]
df = df.drop("Year", axis=1)
df = pd.get_dummies(df, columns=["Brand", "Transmission", "Fuel Type", "Condition"], drop_first=True)
df = df[(df["Price"] >= 7000) & (df["Price"] <= 90000)]
df = df.dropna()

# Split into features (X) and target (y)
X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])

# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing Pipeline (Handle categorical, numerical features, and missing data)
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Define a custom transformer for one-hot encoding
class PandasDummies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Fit method does nothing but is required for scikit-learn compatibility
        return self
    
    def transform(self, X):
        return pd.get_dummies(X)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())  # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", PandasDummies())  # Use custom transformer for one-hot encoding
])

# Combine all transformations into a single preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Models for comparison
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

# Hyperparameter grid for RandomForest and DecisionTree
param_grid = {
    "Random Forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
    },
    "Decision Tree": {
        "model__max_depth": [5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
    },
    "Linear Regression": {
        # Linear regression doesn't require hyperparameter tuning
    }
}

# Store results
results = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid.get(name, {}), cv=5, n_jobs=-1, scoring="neg_mean_absolute_error")
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best model and parameters
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    
    # Make predictions with the best model
    y_pred = np.expm1(grid_search.best_estimator_.predict(X_test))
    
    # Calculate MAE and R²
    mae = mean_absolute_error(np.expm1(y_test), y_pred)
    r2 = r2_score(np.expm1(y_test), y_pred)
    
    print(f"{name} - Mean Absolute Error: {mae:.2f}")
    print(f"{name} - R² Score: {r2:.2f}")
    
    results[name] = {"MAE": mae, "R2": r2}
    
    # Plot results
    plt.figure(figsize=(6, 6))
    plt.scatter(np.expm1(y_test), y_pred, alpha=0.5)
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