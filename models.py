from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
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


def register_model():
    local_runs = mlflow.search_runs()
    if local_runs.size == 0:
        raise RuntimeError('No local runs found')
    mlflow.set_tracking_uri(os.environ['tracking'])
    client = MlflowClient()

    # TODO: experiments를 default 쓰면 ~/0/밑에 생성되고 왜 새로 만들면 그냥 생기냐
    # exp = client.get_experiment_by_name('SMART')
    # if exp is None:
    #     client.create_experiment('SMART', os.environ['ftp'])
    #     exp = client.get_experiment_by_name('SMART')
    
    for idx in range(local_runs.shape[0]):
        batch = get_batch(local_runs.loc[idx])
        try:
            already_registered_model = client.get_registered_model(batch['name'])
            print(f'{batch["name"]} registered')
        except Exception as e:
            # RESOURCE_DOES_NOT_EXISTS
            already_registered_model = None
        if already_registered_model:
            continue
        with mlflow.start_run() as run:
            client.log_batch(run.info.run_id, metrics=batch['metrics'], params=batch['params'], tags=batch['tags'])
            client.log_artifacts(run.info.run_id, os.path.join('./mlruns', '0', local_runs.loc[idx]['run_id'], 'artifacts'))
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


def get_matrix(y_test, y_prediction):
    return {
        'accuracy': accuracy_score(y_test, y_prediction),
        'precision': precision_score(y_test, y_prediction),
        'recall': recall_score(y_test, y_prediction),
        'f1': f1_score(y_test, y_prediction)
    }


class AMPMModel:

    MODEL = 'model'

    def __init__(self, data, params):
        # pandas.DataFrame
        self.data = self.preprocess_data(data)
        self.model = self.create_model(params)
        self.name = self.model.__class__.__name__
        self.params = params

    def preprocess_data(self, data):
        train, test = self.fill(data)
        features = train.columns[5:]
        label = train.columns[4]
        x_train = train[features]
        y_train = train[label]
        x_test = test[features]
        y_test = test[label]
        x_train, x_test = self.encoding(x_train, x_test)
        return (x_train, y_train), (x_test, y_test)

    def fill(self, data):
        train, test = data
        train = train.fillna(0)
        test = test.fillna(0)
        return train, test

    def encoding(self, train, test):
        return train, test

    def create_model(self, params):
        raise AttributeError('Model not defined')

    def optimize(self):
        print(f"running {self.name}...")
        (x_train, y_train), (x_test, y_test) = self.data
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, cv=10, refit=True, scoring="recall_micro")
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        prediction = best_model.predict(x_test)
        print(f"{self.name} : {confusion_matrix(y_test, prediction)}")
        return {
            'name': self.name,
            'model': best_model,
            'metrics': get_matrix(y_test, prediction),
            'params': best_model.get_params()
        }


class RF(AMPMModel):

    def create_model(self, params):
        return RandomForestClassifier()


class XGB(AMPMModel):

    def fill(self, data):
        train, test = data
        train = train.fillna(train.mean())
        train = train.fillna(0)
        test = test.fillna(0)
        return train, test

    def create_model(self, params):
        return XGBClassifier()


class OCSVM(AMPMModel):

    def encoding(self, x):
        return PCA(n_components=1).fit_transform(StandardScaler().fit_transform(x))

    def encoding(self, train, test):
        pca = PCA(n_components=1)
        scaler = StandardScaler()
        return pca.fit_transform(scaler.fit_transform(train)), pca.fit_transform(scaler.fit_transform(test))

    def fill(self, data):
        train, test = data
        train = train.fillna(0)
        test = test.fillna(0)
        train = train[train['failure'] == 0]
        return train, test

    def create_model(self, params):
        return OneClassSVM()