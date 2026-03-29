import asyncio
import threading
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace
import time
import pandas as pd
from typing import Any
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app, Counter, Histogram, Info

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


MODEL = Model()

TIME_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, float("inf"))

REQUESTS_TOTAL = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['endpoint'], buckets=TIME_BUCKETS)

PREPROCESS_LATENCY = Histogram('data_preprocess_duration_seconds', 'Time spent preprocessing data', buckets=TIME_BUCKETS)
FEATURE_VALUES = Histogram('feature_values', 'Values of numerical features', ['feature_name'])
FEATURE_CATEGORICAL = Counter('feature_categorical_total', 'Counts of categorical feature values', ['feature_name', 'category_value'])

INFERENCE_LATENCY = Histogram('model_inference_duration_seconds', 'Time spent in model.predict_proba', buckets=TIME_BUCKETS)
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions by class', ['class_label'])
MODEL_PROBABILITIES = Histogram('model_probabilities', 'Distribution of prediction probabilities')

MODEL_INFO = Info('current_model', 'Information about the currently loaded model')
MODEL_UPDATES = Counter('model_updates_total', 'Total model updates', ['status'])

DATA_LOCK = threading.Lock()
REFERENCE_DATA = []
CURRENT_DATA = []
EVIDENTLY_URL = 'http://158.160.2.37:8000/'
EVIDENTLY_PROJECT_ID = '019d3afe-3198-7a78-b501-032ef27b9191'

def build_and_send_evidently_report(ref_df: pd.DataFrame, curr_df: pd.DataFrame):
    try:
        report = Report([DataDriftPreset()])
        my_eval = report.run(reference_data=ref_df, current_data=curr_df)
        
        workspace = RemoteWorkspace(EVIDENTLY_URL)
        workspace.add_run(EVIDENTLY_PROJECT_ID, my_eval, include_data=False)
        print(f"отчёт в evidently отправлен")
    except Exception as e:
        print(f"ошибка при отправке в evidently отчёта: {e}")

async def evidently_background_task():
    global REFERENCE_DATA, CURRENT_DATA
    
    while True:
        await asyncio.sleep(60)
        
        with DATA_LOCK:
            print(f"проверяем есть ли данные для отчёта")
            if len(CURRENT_DATA) < 10: 
                continue
                
            if not REFERENCE_DATA:
                REFERENCE_DATA.extend(CURRENT_DATA)
                CURRENT_DATA.clear()
                print(f"обновляем референс для отчёта")
                continue
                
            ref_df = pd.DataFrame(REFERENCE_DATA)
            curr_df = pd.DataFrame(CURRENT_DATA)
            CURRENT_DATA.clear()
            
        await asyncio.to_thread(build_and_send_evidently_report, ref_df, curr_df)


def get_model_type(model_obj) -> str:
    if not model_obj:
        return "Unknown"
    
    if hasattr(model_obj, 'steps'):
        return type(model_obj.steps[-1][1]).__name__
        
    return type(model_obj).__name__

def update_model_info():
    model_state = MODEL.get()
    if model_state.run_id and model_state.model:
        model_type = get_model_type(model_state.model)
        MODEL_INFO.info({
            'run_id': model_state.run_id, 
            'features': ','.join(MODEL.features),
            'model_type': model_type
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    update_model_info()

    evidently_task = asyncio.create_task(evidently_background_task())

    yield

    evidently_task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get('/health')
    def health() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(method='GET', endpoint='/health', http_status='200').inc()
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        start_time = time.perf_counter()
        
        model = MODEL.get().model
        if model is None:
            REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', http_status='503').inc()
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        preprocess_start = time.perf_counter()
        df = to_dataframe(request, needed_columns=MODEL.features)
        PREPROCESS_LATENCY.observe(time.perf_counter() - preprocess_start)

        if df.isnull().values.any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', http_status='422').inc()
            raise HTTPException(
                status_code=422, 
                detail=f"Missing or invalid required features for this model: {missing_cols}"
            )

        # лог фичей
        for col in df.columns:
            val = df.iloc[0][col]
            if pd.isna(val):
                continue
                
            if isinstance(val, (int, float, np.number)):
                FEATURE_VALUES.labels(feature_name=col).observe(float(val))
            elif isinstance(val, str):
                FEATURE_CATEGORICAL.labels(feature_name=col, category_value=val).inc()

        # собственно инференс
        try:
            inference_start = time.perf_counter()
            probability = float(model.predict_proba(df)[0][1])
            prediction = int(probability >= 0.5)
            INFERENCE_LATENCY.observe(time.perf_counter() - inference_start)
            
            MODEL_PROBABILITIES.observe(probability)
            MODEL_PREDICTIONS.labels(class_label=str(prediction)).inc()
            
        except Exception as e:
            REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', http_status='400').inc()
            raise HTTPException(status_code=400, detail=f"Model inference error: {str(e)}")

        REQUESTS_TOTAL.labels(method='POST', endpoint='/predict', http_status='200').inc()
        REQUEST_LATENCY.labels(endpoint='/predict').observe(time.perf_counter() - start_time)

        row_dict = df.iloc[0].to_dict()
        row_dict['prediction'] = prediction
        with DATA_LOCK:
            CURRENT_DATA.append(row_dict)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
            MODEL_UPDATES.labels(status='success').inc()
            REQUESTS_TOTAL.labels(method='POST', endpoint='/updateModel', http_status='200').inc()
            update_model_info()
        except Exception as e:
            MODEL_UPDATES.labels(status='failed').inc()
            REQUESTS_TOTAL.labels(method='POST', endpoint='/updateModel', http_status='400').inc()
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to load model with run_id {run_id}. Error: {str(e)}"
            )
            
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
