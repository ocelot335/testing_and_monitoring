import pytest
from fastapi.testclient import TestClient
import pandas as pd
from unittest.mock import patch

from ml_service.app import create_app
from ml_service.schemas import PredictRequest
from ml_service.features import to_dataframe

class DummyModel:
    feature_names_in_ = ['age', 'workclass']
    
    def predict_proba(self, df):
        return [[0.2, 0.8]] 

@pytest.fixture
def client():
    app = create_app()
    
    with patch('ml_service.config.default_run_id', return_value="dummy_id"), \
         patch('ml_service.config.tracking_uri', return_value=""), \
         patch('ml_service.model.load_model', return_value=DummyModel()):
        
        with TestClient(app) as test_client:
            yield test_client

def test_to_dataframe():
    req = PredictRequest(age=30, workclass="Private", education="Bachelors")
    df = to_dataframe(req, needed_columns=["age", "workclass"])
    
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["age", "workclass"]
    assert df.iloc[0]["age"] == 30

def test_predict_success(client):
    payload = {"age": 30, "workclass": "Private"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.8

def test_predict_missing_features(client):
    payload = {"age": 30} 
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
    assert "Missing or invalid required features" in response.json()["detail"]

def test_update_model_invalid_run_id(client):
    with patch('ml_service.model.load_model', side_effect=Exception("MLflow API Error")):
        response = client.post("/updateModel", json={"run_id": "bad_id"})
        assert response.status_code == 400
        assert "Failed to load model with run_id bad_id" in response.json()["detail"]

def test_update_model_success(client):
    response = client.post("/updateModel", json={"run_id": "new_awesome_id"})
    
    assert response.status_code == 200
    assert response.json()["run_id"] == "new_awesome_id"