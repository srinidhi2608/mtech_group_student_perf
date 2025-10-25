import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path="simple_student_dataset.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place simple_student_dataset.csv at repo root.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def preprocess(df):
    df = df.copy()
    feature_cols = ['PUC_Score','Sem_1','Sem_2','Sem_3','Sem_4','Sem_5','Attendance_%','Study_Hours_Per_Day','Sleep_Hours_Per_Day']
    # ensure numeric columns
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['Courses_Completed'] = df['Courses_Completed'].fillna('').astype(str)
    courses_list = df['Courses_Completed'].apply(lambda s: [x.strip().lower() for x in s.split(',') if x.strip()])
    mlb = MultiLabelBinarizer()
    courses_mh = mlb.fit_transform(courses_list)
    courses_cols = [f"course_{c}" for c in mlb.classes_]
    courses_df = pd.DataFrame(courses_mh, columns=courses_cols, index=df.index)
    X = pd.concat([df[feature_cols].reset_index(drop=True), courses_df.reset_index(drop=True)], axis=1)
    X = X.fillna(0)
    y = pd.to_numeric(df['Sem_6'], errors='coerce').fillna(0)
    return X, y, mlb

def train(path="simple_student_dataset.csv", save_prefix="sem6_pred", test_size=0.2, random_state=42):
    df = load_data(path)
    X, y, mlb = preprocess(df)
    if X.shape[0] < 5:
        raise ValueError("Not enough rows to train; need at least 5 rows.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1000)
    artifact = {'model': model, 'mlb': mlb, 'feature_cols': X.columns.tolist()}
    joblib.dump(artifact, f"{MODEL_DIR}/{save_prefix}.joblib")
    preds = model.predict(X_val)
    rmse =  np.sqrt(np.mean((y_val - preds) ** 2))

    print(f"Saved model to {MODEL_DIR}/{save_prefix}.joblib. Validation RMSE: {rmse:.4f}")
    return artifact

if __name__ == '__main__':
    train()