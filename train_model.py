# Refined train_model.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# Scikit-learn modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, make_scorer, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

# Imbalanced-learn modules
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

######################################
#            CONFIGURATION           #
######################################

# Path to the preprocessed CSV (could be on S3 or local)
PREPROCESSED_PATH = "s3://my-bucket/folder/diabetic_data_cleaned.csv"

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
    """
    A custom classifier that wraps a base_estimator and applies a threshold
    to the positive_class probability for predictions. Now includes classes_
    and _estimator_type attributes required by scikit-learn tools.
    """
    def __init__(self, base_estimator, threshold=0.5, positive_class='<30'):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.positive_class = positive_class
        
        # Helps scikit-learn recognize this is a classifier
        self._estimator_type = "classifier"

    def fit(self, X, y):
        # Fit the base estimator
        self.base_estimator.fit(X, y)
        # Store the classes_ attribute so that scikit-learn utilities can find it
        self.classes_ = self.base_estimator.classes_
        # print("ThresholdClassifier classes_:", self.classes_)  # Debug print
        return self

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)

    def predict(self, X):
        # Find the index of the "positive" class
        pos_index = list(self.classes_).index(self.positive_class)
        probs = self.predict_proba(X)[:, pos_index]
        # The other class (we assume it's a binary classification scenario)
        other_class = [c for c in self.classes_ if c != self.positive_class][0]
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
    # Base classifier and threshold
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    threshold_clf = ThresholdClassifier(base_estimator=rf, threshold=0.5, positive_class='<30')

    pipeline = ImbPipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', threshold_clf)
    ])

    # Define parameter grid for GridSearchCV
    # NOTE: param names must follow the pipeline steps + model
    param_grid = {
        # RandomForest hyperparams via pipeline -> 'clf__base_estimator__'
        'clf__base_estimator__n_estimators': [50, 100, 200],
        'clf__base_estimator__max_depth': [5, 10],
        # class_weight can help with imbalance
        'clf__base_estimator__class_weight': [{'<30': 5, 'Not <30': 1},  # 5:1 ratio
                                              {'<30': 3, 'Not <30': 1},  # 3:1 ratio
                                              {'<30': 2, 'Not <30': 1},  # 2:1 ratio
                                              'balanced',                # scikit-learn auto-balancing
                                              None                       # baseline (no weighting)
                                              ],
        # Potentially add more, like min_samples_leaf, threshold etc.
        #threhold
        'clf__threshold': [0.2, 0.3, 0.4, 0.45, 0.5]

    }

    # We'll optimize for F1 specifically for <30 class
    # We can define a custom scorer or use 'f1' (which is default for binary) 
    # but for a multi-class or custom label, we'd define a custom scorer.
    # For demonstration, let's use 'f1_macro' or 'f1_weighted' as a simpler approach.
    # If we want exactly the f1 on <30 specifically, we might define a custom scorer.
    # from sklearn.metrics import make_scorer
    def custom_f1_for_lt30(y_true, y_pred):
        # Evaluate only on <30
        rpt = classification_report(y_true, y_pred, output_dict=True)
        return rpt['<30']['f1-score']  # might fail if <30 wasn't predicted at all
    custom_scorer = make_scorer(custom_f1_for_lt30, greater_is_better=True)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=custom_scorer,
        cv=3,  # 3-fold cross-validation
        n_jobs=-1
    )

    # 4. Start an MLflow experiment/run
    mlflow.set_experiment("MediWatch-Training")
    with mlflow.start_run(run_name="GridSearch_Run") as parent_run: 
        print("Running GridSearchCV ...")
        gs.fit(X_train, y_train)

        # For each param combo, open a brand-new child run
        # removing `run_id=parent_run.info.run_id` so that MLflow
        # actually creates a separate run ID for each combo.
        for i, combo in enumerate(gs.cv_results_["params"]):
            mean_test_score = gs.cv_results_["mean_test_score"][i]
            std_test_score = gs.cv_results_["std_test_score"][i]

            # Each iteration uses nested=True but NO explicit run_id override
            with mlflow.start_run(nested=True): 
                mlflow.log_params(combo)
                mlflow.log_metric("mean_cv_f1_lt30", mean_test_score)
                mlflow.log_metric("std_cv_f1_lt30", std_test_score)
        
        # Best estimator from grid search
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        print(f"Best Params from GridSearchCV: {best_params}")

        # Evaluate best_model on val/test
        y_val_pred = best_model.predict(X_val)
        val_report = classification_report(y_val, y_val_pred, digits=4, output_dict=True)        

        y_test_pred = best_model.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)        

        # Extended metrics
        # 1. Extract f1, precision & recall for <30
        # After computing val_report and test_report
        val_f1_lt30 = val_report['<30']['f1-score']
        val_precision_lt30 = val_report['<30']['precision']
        val_recall_lt30 = val_report['<30']['recall']

        test_f1_lt30 = test_report['<30']['f1-score']
        test_precision_lt30 = test_report['<30']['precision']
        test_recall_lt30 = test_report['<30']['recall']


        # 2. Log these additional metrics to MLflow
        mlflow.log_metric("val_f1_lt30", val_f1_lt30)
        mlflow.log_metric("val_precision_lt30", val_precision_lt30)
        mlflow.log_metric("val_recall_lt30", val_recall_lt30)

        mlflow.log_metric("test_f1_lt30", test_f1_lt30)
        mlflow.log_metric("test_precision_lt30", test_precision_lt30)
        mlflow.log_metric("test_recall_lt30", test_recall_lt30)

        # 3. Create & log confusion matrices (numeric)
        
        # For validation
        y_val_labels = ["<30", "Not <30"]  # adjust if your actual labels differ
        val_cm = confusion_matrix(y_val, y_val_pred, labels=y_val_labels)
        # val_cm is a 2x2 array, e.g.:
        # [[TP_30, FN_30],
        #  [FP_30, TN_30]]

        # Flatten if you want each cell
        # Or just log them as separate metrics:
        mlflow.log_metric("val_cm_lt30_lt30", val_cm[0, 0]) # True <30
        mlflow.log_metric("val_cm_lt30_not_lt30", val_cm[0, 1])
        mlflow.log_metric("val_cm_not_lt30_lt30", val_cm[1, 0])
        mlflow.log_metric("val_cm_not_lt30_not_lt30", val_cm[1, 1])
        # For test
        test_cm = confusion_matrix(y_test, y_test_pred, labels=y_val_labels)
        mlflow.log_metric("test_cm_lt30_lt30", test_cm[0, 0])
        mlflow.log_metric("test_cm_lt30_not_lt30", test_cm[0, 1])
        mlflow.log_metric("test_cm_not_lt30_lt30", test_cm[1, 0])
        mlflow.log_metric("test_cm_not_lt30_not_lt30", test_cm[1, 1])

        # 4. Optionally, log confusion matrix as an image artifact
        #    so you can visualize it later in MLflow UI
        

        # Validation CM image
        fig_val, ax_val = plt.subplots(figsize=(4,4))
        sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", ax=ax_val,
                    xticklabels=y_val_labels, yticklabels=y_val_labels)
        ax_val.set_xlabel("Predicted")
        ax_val.set_ylabel("Actual")
        ax_val.set_title("Validation Confusion Matrix")

        val_cm_path = "val_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(val_cm_path)
        mlflow.log_artifact(val_cm_path)
        plt.close(fig_val)  # close the figure to free memory

        # Test CM image
        fig_test, ax_test = plt.subplots(figsize=(4,4))
        sns.heatmap(test_cm, annot=True, fmt="d", cmap="Greens", ax=ax_test,
                    xticklabels=y_val_labels, yticklabels=y_val_labels)
        ax_test.set_xlabel("Predicted")
        ax_test.set_ylabel("Actual")
        ax_test.set_title("Test Confusion Matrix")

        test_cm_path = "test_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(test_cm_path)
        mlflow.log_artifact(test_cm_path)
        plt.close(fig_test)

        # End of run
        print(f"Val Precision(<30): {val_precision_lt30:.4f}, Val Recall(<30): {val_recall_lt30:.4f}")
        print(f"Test Precision(<30): {test_precision_lt30:.4f}, Test Recall(<30): {test_recall_lt30:.4f}")

        print("Done with extended metrics logging.")


        # Log final metrics, best params, and model
        mlflow.log_params(best_params)
        mlflow.log_metric("val_f1_lt30", val_f1_lt30)
        mlflow.log_metric("test_f1_lt30", test_f1_lt30)

        # Log the best pipeline
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print("Final Validation F1 (<30):", val_f1_lt30)
        print("Final Test F1 (<30):", test_f1_lt30)

        # Optional: Register the best model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="MediWatchReadmissionModel")

        print("Done with GridSearch + MLflow logging.")

if __name__ == "__main__":
    run_training()