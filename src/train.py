import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna

from src.data_processing import load_students, build_student_feature_pipeline, featurize_students

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_target(df):
    df = df.copy()
    df['gpa'] = df['gpa'].astype(float)
    return df

def objective(trial, X_train, y_train, X_valid, y_valid):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    }
    model = lgb.train(param, dtrain, valid_sets=[dvalid], early_stopping_rounds=50, verbose_eval=False)
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds, squared=False)

def train(student_csv_path, save_prefix='student_perf', n_trials=20):
    df = load_students(student_csv_path)
    df = prepare_target(df)
    preprocessor = build_student_feature_pipeline(df)
    X = featurize_students(preprocessor, df)
    y = df['gpa'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    study = optuna.create_study(direction='minimize')
    func = lambda trial: objective(trial, X_train, y_train, X_valid, y_valid)
    study.optimize(func, n_trials=n_trials)
    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1})
    dtrain = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(best_params, dtrain, num_boost_round=1000)
    joblib.dump(preprocessor, f"{MODEL_DIR}/{save_prefix}_preprocessor.joblib")
    joblib.dump(final_model, f"{MODEL_DIR}/{save_prefix}_lgbm.joblib")
    preds = final_model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    print("Saved model and preprocessor to", MODEL_DIR)
    print(f"Validation RMSE: {rmse:.4f}")
    return final_model, preprocessor

if __name__ == "__main__":
    train("students_sample.csv")
