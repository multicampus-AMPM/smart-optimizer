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
        mlflow.set_tracking_uri(os.environ['tracking'])
        client = MlflowClient()

        # TODO: experiments를 default 쓰면 ~/0/밑에 생성되고 왜 새로 만들면 그냥 생기냐
        # exp = client.get_experiment_by_name('SMART')
        # if exp is None:
        #     client.create_experiment('SMART', os.environ['ftp'])
        #     exp = client.get_experiment_by_name('SMART')    
        
        mlflow.autolog()
        for idx in range(local_runs.shape[0]):
            batch = get_batch(local_runs.loc[idx])
            # with mlflow.start_run(experiment_id=exp.experiment_id) as run:
            with mlflow.start_run() as run:
                client.log_batch(run.info.run_id, metrics=batch['metrics'], params=batch['params'], tags=batch['tags'])
                client.log_artifacts(run.info.run_id, os.path.join('/home/mlruns', '0', local_runs.loc[idx]['run_id'], 'artifacts'))
        
                # TODO: 기준을 무엇으로 할런지
                if idx == 0:
                    client.create_registered_model('smart-model')
                    # source 위치 반드시 실제 저장소랑 똑같이 맞춰야함 
                    # source 기준으로 실제 파일 접근함
                    client.create_model_version(
                        name="smart-model",
                        source=f"{os.environ['ftp']}/0/{run.info.run_id}/artifacts/model",
                        run_id=run.info.run_id,
                        description='model for smart',
                        tags={'test' : 'yes'}
                    )
                    client.transition_model_version_stage(
                        name="smart-model",
                        version=1,
                        stage="Production"
                    )


def parse_env():
    # use get function instead of key access to avoid error
    repo = os.environ.get('repo')

    # TODO regex validation
    if repo is None:
        os.environ['repo'] = 'model-repository'
    os.environ['tracking'] = f'http://{os.environ["repo"]}:5000'
    os.environ['ftp'] = f'ftp://mlflow:mlflow@{os.environ["repo"]}/mlflow/artifacts/'


if __name__ == '__main__':
    parse_env()
    save_runs()