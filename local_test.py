from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, root_mean_squared_error

from model import CNN, CNNSpace,Model

folder = Path(__file__).resolve().parent / "data"
print(f"Using data folder: {folder}")

if __name__ == "__main__":
    # Load the dataset
    target_column = 'TARGET_CTR'
    df = pd.read_csv(folder / 'CTR_dataset_modified.csv')
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=42, shuffle=False)
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    y_train.fillna(0, inplace=True)  # Fill NaN values with 0
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]    

    # Create a model instance
    model = Model(dataset_name='dataset1')
    model.fit(X_train.values, y_train.values, override=False)

    # Make predictions
    predictions = model.predict(X_test.fillna(0).values)

    # Calculate RMSE
    rmse = root_mean_squared_error(y_test.fillna(0).values, predictions)
    print(f"RMSE: {rmse} for CTR prediction")

    target_column = 'TARGET_CONV'
    df = pd.read_csv(folder / 'CONV_dataset_modified.csv')
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=42, shuffle=False)
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]
    y_train.fillna(0, inplace=True)  # Fill NaN values with 0
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]    

    # Create a model instance
    model = Model(dataset_name='dataset2')
    model.fit(X_train.values, y_train.values, override=False)

    # Make predictions
    predictions = model.predict(X_test.fillna(0).values)

    # Calculate RMSE
    rmse = root_mean_squared_error(y_test.fillna(0).values, predictions)
    print(f"RMSE: {rmse} for conv prediction")