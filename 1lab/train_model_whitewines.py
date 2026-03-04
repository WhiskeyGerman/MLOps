import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def scale_frame(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, pd.Series, StandardScaler]:
    df = frame.copy()
    x_data = df.drop(columns=["quality"])
    y_data = df["quality"]

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x_data.values)

    return x_scale, y_data, scaler


def eval_metrics(
    actual: np.ndarray, pred: np.ndarray
) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2


def train() -> None:
    df = pd.read_csv("./df_clear.csv")
    x_data, y_data, scaler = scale_frame(df)

    x_train, x_val, y_train, y_val = train_test_split(
        x_data,
        y_data,
        test_size=0.3,
        random_state=42,
    )

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.001, 0.05, 0.01, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "fit_intercept": [False, True],
    }

    mlflow.set_experiment("wine_quality_model")

    with mlflow.start_run():
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=-1)
        
        clf.fit(x_train, y_train.values.reshape(-1))

        best_model = clf.best_estimator_
        y_pred = best_model.predict(x_val)

        rmse, mae, r2 = eval_metrics(y_val.values, y_pred)

        mlflow.log_param("alpha", best_model.alpha)
        mlflow.log_param("l1_ratio", best_model.l1_ratio)
        mlflow.log_param("penalty", best_model.penalty)
        mlflow.log_param("loss", best_model.loss)
        mlflow.log_param("fit_intercept", best_model.fit_intercept)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best_model.predict(x_train)
        signature = infer_signature(x_train, predictions)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        with open("sgd_wine.pkl", "wb") as file:
            joblib.dump(best_model, file)
        
        with open("scaler_wine.pkl", "wb") as file:
            joblib.dump(scaler, file)
