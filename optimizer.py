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
            # params.append(Param(key, dict_info[key]))
            # 역직렬화에서 None을 literal하게 None으로 치환하면 sqlite에서 뱉어냄 (버그인듯)
            params.append(Param(key, 'None' if dict_info[key] is None else dict_info[key]))
        elif key.startswith('tags'):
            # 5000자 넘어가면 저장 못함
            if key == 'tags.mlflow.log-model.history':
                continue
            tags.append(RunTag(key, dict_info[key]))
    return {
        'name' : dict_info['tags.estimator_name'],
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
            print(batch)
            with mlflow.start_run() as run:
                client.log_batch(run.info.run_id, metrics=batch['metrics'], params=batch['params'], tags=batch['tags'])
                client.log_artifacts(run.info.run_id, os.path.join('/home/mlruns', '0', local_runs.loc[idx]['run_id'], 'artifacts'))                
                client.create_registered_model(batch['name'])
                # source 위치 반드시 실제 저장소랑 똑같이 맞춰야함 
                # source 기준으로 실제 파일 접근함
                client.create_model_version(
                    name=batch['name'],
                    source=f"{os.environ['ftp']}/0/{run.info.run_id}/artifacts/model",
                    run_id=run.info.run_id,
                    description=f"{batch['name']} model for smart"
                )
                client.transition_model_version_stage(
                    name=batch['name'],
                    version=1,
                    stage="Production"
                )


def parse_env():
    # use get function instead of key access to avoid error
    repo = os.environ.get('repo')
    mode = os.environ.get('MODE')

    # TODO regex validation
    if repo is None:
        os.environ['repo'] = 'model-repository'
    if mode is None:
        os.environ['MODE'] = 'none'
    os.environ['tracking'] = f'http://{os.environ["repo"]}:5000'
    os.environ['ftp'] = f'ftp://mlflow:mlflow@{os.environ["repo"]}/mlflow/artifacts/'


if __name__ == '__main__':
    parse_env()
    # if os.environ['MODE'] == 'install':
        # save_runs()
    save_runs()