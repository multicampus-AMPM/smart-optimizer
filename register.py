import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
import os
import time


def get_batch(run_info):
    metrics = list()
    params = list()
    tags = list()
    dict_info = run_info.to_dict()
    for key in dict_info.keys():
        # 역직렬화에서 None을 literal하게 NoneType으로 치환하면 sqlite에서 뱉어냄 (버그인듯)
        value = 'None' if dict_info[key] is None else dict_info[key]
        if key.startswith('metrics'):
            metrics.append(Metric(key.replace('metrics.', ''), value, int(time.time() * 1000), 0))
        elif key.startswith('params'):          
            params.append(Param(key.replace('params.', ''), value))
        elif key.startswith('tags'):
            # 5000자 넘어가면 저장 못함
            if key == 'tags.mlflow.log-model.history':
                continue
            tags.append(RunTag(key.replace('tags.', ''), value))
    return {
        'name' : dict_info['tags.estimator_name'],
        'metrics' : metrics,
        'params' : params,
        'tags' : tags
    }


def log_model():
    local_runs = mlflow.search_runs()
    if local_runs.size == 0:
        raise RuntimeError('No local runs found')
    
    mlflow.set_tracking_uri(os.environ['tracking'])
    client = MlflowClient()
    best_model = local_runs[(local_runs['metrics.recall'] == local_runs['metrics.recall'].max()) & (local_runs['tags.estimator_name'] != 'OneClassSVM')]['tags.estimator_name'].iloc[0]
    for idx in range(local_runs.shape[0]):
        batch = get_batch(local_runs.loc[idx])
        with mlflow.start_run() as run:
            client.log_batch(run.info.run_id, metrics=batch['metrics'], params=batch['params'], tags=batch['tags'])
            client.log_artifacts(run.info.run_id, os.path.join('./mlruns', '0', local_runs.loc[idx]['run_id'], 'artifacts'))
            uri = f"{os.environ['ftp']}/0/{run.info.run_id}/artifacts/model"
            register_model(client, batch['name'], uri, run.info.run_id, '1' if batch['name'] == best_model else '0')

    # recall 기준으로 best model 등록
    # best_run = mlflow.search_runs(["0"], order_by=["metrics.recall DESC"], max_results=1)
    # remote_runs = mlflow.search_runs(["0"], order_by=["metrics.recall DESC"])
    # if not remote_runs.empty:
    #     for i in range(1, remote_runs.shape[0]):
    #         info = remote_runs.iloc[i].T
    #         register_model(client, info['tags.estimator_name'], info['artifact_uri'], info['run_id'], '1' if i == 0 else '0')


def register_model(client, model_name, source, run_id, best):
    try:
        test = client.get_registered_model(model_name)
    except Exception as e:
        # RESOURCE_DOES_NOT_EXIST
        test = None
    # 모델이 이미 등록되어 있으면 아무 것도 안함
    if test is None:
        client.create_registered_model(model_name)
        # source 위치 반드시 실제 저장소랑 똑같이 맞춰야함
        # source 기준으로 실제 파일 접근함
        client.set_registered_model_tag(model_name, 'best', best)
        client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
            description=f"estimator '{model_name}' for S.M.A.R.T."
        )
        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Production"
        )


def get_exp(client, name):
    exp = client.get_experiment_by_name(name)
    if exp is None:
        client.create_experiment(name, os.environ['ftp'])
        exp = client.get_experiment_by_name(name)
    return exp