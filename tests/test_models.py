"""
test_models.py
--------------
Tests for model training, prediction, evaluation, and persistence.
"""

import numpy as np
import torch
import joblib

from src.models.baseline import BaselineModel
from src.models.mlp import MaliciousPDFClassifier, train_mlp, PDFDataset
from src.models.evaluator import evaluate_model

def test_baseline_model_train_predict(dummy_feature_matrix):
    X, y = dummy_feature_matrix
    params = {"n_estimators": [10], "max_depth": [3]}
    model = BaselineModel("random_forest", params=params)
    
    # Train
    model.train(X, y, cv_folds=2)
    assert model.best_estimator_ is not None
    
    # Predict
    preds = model.predict(X)
    assert preds.shape == (10,)
    
    probs = model.predict_proba(X)
    assert probs.shape == (10, 2)

def test_mlp_forward_pass():
    model = MaliciousPDFClassifier()
    x = torch.randn(5, 37)
    out = model(x)
    assert out.shape == (5, 1)

def test_mlp_training_loop(dummy_feature_matrix, tmp_path):
    X, y = dummy_feature_matrix
    dataset = PDFDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5)
    
    model = MaliciousPDFClassifier()
    save_path = tmp_path / "mlp_best.pt"
    
    history = train_mlp(
        model, 
        train_loader=loader, 
        val_loader=loader, 
        max_epochs=2, 
        save_path=save_path
    )
    assert "train_loss" in history
    assert len(history["train_loss"]) == 2
    assert save_path.exists()

def test_model_save_load_roundtrip(dummy_feature_matrix, tmp_path):
    X, y = dummy_feature_matrix
    model = BaselineModel("random_forest", params={"n_estimators": [10]})
    model.train(X, y, cv_folds=2)
    
    save_path = tmp_path / "rf_test.pkl"
    model.save(save_path)
    
    loaded_model = BaselineModel.load(save_path)
    assert loaded_model.display_name == "Random Forest"
    
    preds_orig = model.predict(X)
    preds_loaded = loaded_model.predict(X)
    assert np.array_equal(preds_orig, preds_loaded)

def test_evaluate_model(dummy_feature_matrix):
    X, y = dummy_feature_matrix
    model = BaselineModel("random_forest", params={"n_estimators": [10]})
    model.train(X, y, cv_folds=2)
    
    # Use 2 classes so evaluate_model doesn't complain
    y[0] = 0
    y[1] = 1
    
    results = evaluate_model(model, X, y, "test_rf")
    assert "accuracy" in results
    assert "f1" in results
    assert "precision" in results
    assert "recall" in results
    assert "auc_roc" in results

def test_training_reproducibility(dummy_feature_matrix):
    # NFR-204
    X, y = dummy_feature_matrix
    params = {"n_estimators": [10]}
    
    model1 = BaselineModel("random_forest", params=params)
    model1.train(X, y, cv_folds=2)
    preds1 = model1.predict(X)
    
    model2 = BaselineModel("random_forest", params=params)
    model2.train(X, y, cv_folds=2)
    preds2 = model2.predict(X)
    
    assert np.array_equal(preds1, preds2)

def test_no_raw_data_in_saved_model(dummy_feature_matrix, tmp_path):
    # SEC-08: Check that raw training samples are not inside the joblib file
    X, y = dummy_feature_matrix
    model = BaselineModel("random_forest", params={"n_estimators": [10]})
    model.train(X, y, cv_folds=2)
    
    save_path = tmp_path / "sec_test.pkl"
    model.save(save_path)
    
    saved_data = joblib.load(save_path)
    # Check that there's no large array corresponding to training data
    for key, value in saved_data.items():
        if isinstance(value, np.ndarray):
            assert value.shape != X.shape
