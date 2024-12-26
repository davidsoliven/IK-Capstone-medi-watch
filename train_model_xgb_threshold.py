# train_model_xgb_threshold.py

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, make_scorer, precision_recall_curve, auc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

######################################
# 1) A small label map for y
######################################
label_map = {'<30': 1, 'Not <30': 0}
inv_label_map = {1: '<30', 0: 'Not <30'}

######################################
# 2) ThresholdClassifier that expects y in {0,1}
######################################
class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, threshold=0.5):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self._estimator_type = "classifier"
        # We'll define self.classes_ after fit => ['<30','Not <30']


    def fit(self, X, y_str):
        """
        Converts y_str (which is ['<30','Not <30']) to numeric (1 or 0),
        fits base_estimator on that numeric y, and then resets self.classes_
        to the original string labels.
        """
        # 1) Map strings to numeric
        if y_str.dtype == object:  # Check if input is string-based
            y_num = np.array([label_map[label] for label in y_str])
        else:
            y_num = y_str  # Assume it's already numeric

        # 2) Fit the underlying XGBoost or other classifier
        self.base_estimator.fit(X, y_num)

        # 3) We'll keep classes_ in string form, for scikit-learn compatibility
        #    scikit-learn will see that we have two classes: ['<30','Not <30']
        self.classes_ = np.array(['<30', 'Not <30'], dtype=object)
        return self

    def predict_proba(self, X):
        """
        Returns probability for the numeric class 1 = <30, class 0 = Not <30
        in the shape (n_samples, 2).
        The second column is the prob of <30 (class 1).
        """
        check_is_fitted(self, 'classes_')
        # XGB predict_proba => shape [n_samples, 2], columns = [prob class0, prob class1].
        # We'll interpret column 1 as <30
        raw_proba = self.base_estimator.predict_proba(X)

        # Reorder columns so that col 0 => <30, col 1 => Not <30 if you want to match
        # self.classes_ = ['<30','Not <30'] in that order. Let's do that carefully:
        # By default, XGBoost or any numeric-based classifier might say col 0 => class 0, col 1 => class 1
        # We'll define col 0 => prob <30, col 1 => prob Not <30
        # If the classifier by default returns [prob(0), prob(1)], we just swap columns:
        # Because 1 = <30, 0 = Not <30, so col 1 is actually <30, col 0 is Not <30.
        proba_lt30 = raw_proba[:,1]  # class 1
        proba_not_lt30 = raw_proba[:,0]  # class 0
        out_proba = np.stack([proba_lt30, proba_not_lt30], axis=1)
        # shape => (n_samples, 2), column 0 => <30, column 1 => Not <30
        return out_proba

    def predict(self, X):
        check_is_fitted(self, 'classes_')
        out_proba = self.predict_proba(X)
        # out_proba[:, 0] => prob of class 1 => <30
        proba_lt30 = out_proba[:, 0]
        # threshold
        # if > threshold => 1 (<30), else 0 (Not <30)
        numeric_preds = np.where(proba_lt30 > self.threshold, 1, 0)
        return numeric_preds

######################################
# 3) Custom metrics
######################################
def cost_sensitive_metric(y_true, y_pred):
    """
    Example cost: FN=5, FP=1, TP=TN=0
    We'll return negative cost => bigger is better
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    # cm => [[TP,FN],[FP,TN]]
    TP = cm[0,0]
    FN = cm[0,1]
    FP = cm[1,0]
    cost_fn = 5
    cost_fp = 1
    total_cost = FN*cost_fn + FP*cost_fp
    return -total_cost

cost_scorer = make_scorer(cost_sensitive_metric, greater_is_better=True)

# PR AUC for class 1 (which is <30)
def pr_auc_for_class1(y_true, probs_class1):
    """
    We'll compute Precision-Recall AUC for the minority class coded as 1
    """
    precision, recall, _ = precision_recall_curve(y_true, probs_class1)
    return auc(recall, precision)

# Then define a custom scorer:
pr_auc_custom_scorer = make_scorer(pr_auc_for_class1, needs_proba=True, greater_is_better=True)

######################################
# 4) Main training function
######################################
def run_training_xgb_threshold():
    PREPROCESSED_PATH = "s3://mlflow-artifacts-001573566022/diabetic_data_cleaned.csv"
    print(f"Loading data from {PREPROCESSED_PATH} ...")
    df = pd.read_csv(PREPROCESSED_PATH)

    # Convert y to numeric
    y_str = df["readmitted_binary"]  # strings: <30 or Not <30
    y = y_str.map(label_map)         # numeric: 1 or 0
    X = df.drop(columns=["readmitted_binary"])

    # splitted data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # numeric/categorical
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False, sparse_output=False), cat_cols)
    ])

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    threshold_clf = ThresholdClassifier(base_estimator=xgb, threshold=0.5)

    pipeline = ImbPipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),  # if you want to oversample after encoding
        ('clf', threshold_clf)
    ])

    param_grid = {
        'clf__base_estimator__n_estimators': [50,100],
        'clf__base_estimator__max_depth': [3,5],
        'clf__base_estimator__learning_rate': [0.1, 0.01],
        'clf__base_estimator__scale_pos_weight': [1,3,5],
        'clf__threshold': [0.3, 0.5]
    }

    scorers = {"pr_auc_class1": pr_auc_custom_scorer, "cost_scorer": cost_scorer}
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scorers,
        refit="pr_auc_class1",
        cv=3,
        n_jobs=-1
    )
    
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorers,
        refit="pr_auc_class1",  # choose best param set by PR AUC
        cv=3,
        n_jobs=-1
    )

    mlflow.set_experiment("MediWatch-Training-XGB-Threshold-Fixed")
    with mlflow.start_run(run_name="XGB_Threshold_GridSearch"):
        print("Fitting GridSearchCV with XGB + threshold classifier (numeric y).")
        print("y_train unique values:", np.unique(y_train))
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        best_params = gs.best_params_
        print("Best params:", best_params)

        # Evaluate on val/test
        y_val_pred = best_model.predict(X_val)   # 0 or 1
        y_val_prob = best_model.predict_proba(X_val)[:,1]
        val_pr_auc = pr_auc_for_class1(y_val, y_val_prob)
        val_cost = cost_sensitive_metric(y_val, y_val_pred)

        y_test_pred = best_model.predict(X_test)
        y_test_prob = best_model.predict_proba(X_test)[:,1]
        test_pr_auc = pr_auc_for_class1(y_test, y_test_prob)
        test_cost = cost_sensitive_metric(y_test, y_test_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.log_metric("val_cost", val_cost)
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("test_cost", test_cost)

        # log the final pipeline
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"Val PR AUC: {val_pr_auc:.3f}, cost: {val_cost}")
        print(f"Test PR AUC: {test_pr_auc:.3f}, cost: {test_cost}")

        # If you want final predictions in string form, you can invert them:
        # final_pred_str = [inv_label_map[pred] for pred in y_val_pred]

        print("Done with XGB + threshold + numeric Y approach.")