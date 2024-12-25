# Refined train_model.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

# Imbalanced-learn modules
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

######################################
#            CONFIGURATION           #
######################################

# Path to the preprocessed CSV (could be on S3 or local)
PREPROCESSED_PATH = "s3://mlflow-artifacts-001573566022/diabetic_data_cleaned.csv"

# Columns that need one-hot encoding
# You might want to define these in code or dynamically infer them
categorical_cols = [
    "race", "gender", "age", "payer_code", "medical_specialty", "diag_1",
    "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "metformin",
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change",
    "diabetesMed", "admission_type", "discharge_disposition",
    "admission_source"
]

######################################
#       Custom Threshold Class       #
######################################
class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, threshold=0.5, positive_class='<30'):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.positive_class = positive_class
    
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
    
    def predict(self, X):
        # Find the index of the positive class
        classes = self.base_estimator.classes_
        pos_index = list(classes).index(self.positive_class)
        
        probs = self.predict_proba(X)[:, pos_index]
        # Choose either <30 or Not <30 based on threshold
        other_class = [c for c in classes if c != self.positive_class][0]
        return np.where(probs > self.threshold, self.positive_class, other_class)

######################################
#            Main Training           #
######################################
def run_training():
    """
    Reads the preprocessed data from S3 (or local),
    applies SMOTE + threshold adjustment,
    and logs model + metrics to MLflow.
    """
    # 1. Load Preprocessed Data
    print(f"Loading preprocessed data from {PREPROCESSED_PATH} ...")
    df = pd.read_csv(PREPROCESSED_PATH)

    # Separate features (X) and target (y)
    X = df.drop(columns=["readmitted_binary"])
    y = df["readmitted_binary"]

    # Dynamically determine numeric 
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    

    # 2. Split the data
    print("Splitting data into train, val, test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # 3. Define pipeline steps
    print("Building pipeline with OneHotEncoder, SMOTE, and RandomForest + ThresholdClassifier...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
        ],
        remainder='drop'
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    threshold_clf = ThresholdClassifier(base_estimator=rf, threshold=0.5, positive_class='<30')

    pipeline = ImbPipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', threshold_clf)
    ])

    # 4. Start an MLflow experiment/run
    mlflow.set_experiment("MediWatch-Training")
    with mlflow.start_run(run_name="Train_Run") as run:
        run_id = run.info.run_id
        
        # Fit on the training data
        pipeline.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        val_report = classification_report(y_val, y_val_pred, digits=4, output_dict=True)
        
        # Extract F1 specifically for <30
        val_f1 = val_report['<30']['f1-score']
        print("Validation Set Classification Report:")
        print(classification_report(y_val, y_val_pred, digits=4))
        
        # Log metrics
        mlflow.log_metric("val_f1", val_f1)

        # Optionally, evaluate on test set if you want to log that as well
        y_test_pred = pipeline.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
        test_f1 = test_report['<30']['f1-score']
        mlflow.log_metric("test_f1", test_f1)

        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
        # Add any relevant tags or parameters
        mlflow.set_tag("threshold", 0.5)
        mlflow.set_tag("oversampling", "SMOTE")
        
        print(f"Validation F1: {val_f1}, Test F1: {test_f1}")
        print(f"Run ID: {run_id}")

if __name__ == "__main__":
    run_training()