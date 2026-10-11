"""
TabStress: Model Family Factory
Standardizes the 6 classifier families across tabular benchmarks:
1. CART (DecisionTreeClassifier)
2. Random Forest (RandomForestClassifier)
3. Gradient Boosted Decision Trees:
   - XGBoost (XGBClassifier)
   - LightGBM (LGBMClassifier)
   - CatBoost (CatBoostClassifier)
4. Kernel SVM (RBF-kernel SVC)
5. Deep Tabular MLP (MLPClassifier)
6. K-Nearest Neighbors (KNeighborsClassifier)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def get_model(family_name: str, random_state: int = 42):
    family_name = family_name.lower().replace("-", "").replace("_", "")

    if family_name in ["cart", "decisiontree", "dt"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(random_state=random_state))
        ])

    elif family_name in ["randomforest", "rf"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1))
        ])

    elif family_name in ["xgboost", "xgb"]:
        # XGBoost handles missing values natively, but imputer ensures pipeline stability
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1
            ))
        ])

    elif family_name in ["lightgbm", "lgb"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            ))
        ])

    elif family_name in ["catboost", "cb"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                random_state=random_state,
                verbose=0,
                thread_count=-1
            ))
        ])

    elif family_name in ["svm", "rbfsvm"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=random_state))
        ])

    elif family_name in ["mlp", "neuralnet"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=300,
                early_stopping=True,
                random_state=random_state
            ))
        ])

    elif family_name in ["knn", "kneighbors"]:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
        ])

    else:
        raise ValueError(f"Unknown model family: {family_name}")

def get_all_model_families():
    return ["cart", "randomforest", "xgboost", "lightgbm", "catboost", "rbfsvm", "mlp", "knn"]
