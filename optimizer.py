import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
import time


def get_batch(run_info):
    metrics = list()
    params = list()
    tags = list()
    dict_info = run_info.to_dict()
    for key in dict_info.keys():
        if key.startswith('metric'):
            metrics.append(Metric(key, dict_info[key], int(time.time() * 1000), 0))
        elif key.startswith('params'):
            params.append(Param(key, dict_info[key]))
        elif key.startswith('tags'):
            tags.append(RunTag(key, dict_info[key]))
    return {
        'metrics' : metrics,
        'params' : params,
        'tags' : tags
    }


def save_runs():
    local_runs = mlflow.search_runs()
    if local_runs.size != 0:
        mlflow.set_tracking_uri(f'http://{os.environ["repo"]}:5000')
        client = MlflowClient()

        exp = client.get_experiment_by_name('SMART')
        if exp is None:
            client.create_experiment('SMART', f'ftp://mlflow:mlflow@{os.environ["repo"]}/mlflow/artifacts')
            exp = client.get_experiment_by_name('SMART')    
        
        mlflow.autolog()
        for idx in range(local_runs.shape[0]):
            batch = get_batch(local_runs.loc[idx])
            with mlflow.start_run(experiment_id=exp.experiment_id) as run:
                client.log_batch(run.info.run_id, metrics=batch['metrics'], params=batch['params'], tags=batch['tags'])
                client.log_artifacts(run.info.run_id, os.path.join('/home/mlruns', '0', local_runs.loc[idx]['run_id'], 'artifacts'))
                # TODO 모델을 smart-model란 이름으로 저장하고 stage를 production으로 바꿔야함
                


def register_model(client):
    pass



def parse_env():
    # use get function instead of key access to avoid error
    repo = os.environ.get('repo')

    # TODO regex validation
    if repo is None:
        os.environ['repo'] = 'model-repository'


if __name__ == '__main__':
    parse_env()
    save_runs()