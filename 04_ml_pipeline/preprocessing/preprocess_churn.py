import os
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def create_db_engine():
    """
    Create a PostgreSQL database connection using environment variables.
    """

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    load_dotenv(env_path)

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
    )

    return create_engine(connection_url)


def load_gold_data():
    """
    Load the Gold churn model input table from PostgreSQL.
    """

    engine = create_db_engine()

    query = """
    SELECT *
    FROM analytics_gold.gold_churn_model_input
    """

    df = pd.read_sql(query, engine)

    return df


def prepare_ml_data(df, drop_additional_cols=None):
    """
    Prepare churn data for supervised machine learning.
    """

    df_ml = df.copy()

    # Standardize duplicated categorical values
    df_ml["preferred_payment_mode"] = df_ml["preferred_payment_mode"].replace({
        "cod": "cash on delivery",
        "cc": "credit card"
    })

    df_ml["preferred_login_device"] = df_ml["preferred_login_device"].replace({
        "phone": "mobile phone"
    })

    # Remove metadata columns that should not be used for modeling
    drop_cols = [
        "customer_id",
        "batch_id",
        "loaded_at",
        "loaded_by"
    ]

    if drop_additional_cols is not None:
        drop_cols = drop_cols + drop_additional_cols

    df_ml = df_ml.drop(columns=drop_cols)

    X = df_ml.drop(columns=["churn"])
    y = df_ml["churn"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return X, y, preprocessor, categorical_features, numeric_features


def load_and_prepare_data(test_size=0.2, random_state=42, drop_additional_cols=None):
    """
    Load Gold data, prepare ML features, and split into train/test sets.
    """

    df = load_gold_data()

    X, y, preprocessor, categorical_features, numeric_features = prepare_ml_data(
        df,
        drop_additional_cols=drop_additional_cols
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features
    }