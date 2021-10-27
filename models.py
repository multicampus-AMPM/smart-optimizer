from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
import os
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import numpy as np


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
            test = client.get_registered_model(batch['name'])
        except Exception as e:
            # RESOURCE_DOES_NOT_EXIST
            test = None

        if test is not None:
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
        raise AttributeError('Model not defined')

    def create_model(self, params):
        raise AttributeError('Model not defined')

    def optimize(self):
        print(f"running {self.name}...")
        (x_train, y_train), (x_test, y_test) = self.data
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, cv=10, refit=True, scoring="recall_micro")
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        prediction = best_model.predict(x_test)
        if self.name == 'OneClassSVM' or 'OneClassSVMWithKeras':
            prediction = np.where(prediction == 1, 0, prediction)
            prediction = np.where(prediction == -1, 1, prediction)
        print(f"{self.name} : {confusion_matrix(y_test, prediction)}")
        print(prediction)
        return {
            'name': self.name,
            'model': best_model,
            'metrics': get_matrix(y_test, prediction),
            'params': best_model.get_params()
        }


class RF(AMPMModel):

    def preprocess_data(self, data):
        train, test = data
        train = train.fillna(0)
        test = test.fillna(0)
        features = train.columns[5:]
        label = train.columns[4]
        x_train = train[features]
        y_train = train[label]
        x_test = test[features]
        y_test = test[label]
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return (x_train, y_train), (x_test, y_test)

    def create_model(self, params):
        return RandomForestClassifier()


class XGB(AMPMModel):

    def preprocess_data(self, data):
        train, test = data
        train = train.fillna(0)
        test = test.fillna(0)
        features = train.columns[5:]
        label = train.columns[4]
        x_train = train[features]
        y_train = train[label]
        x_test = test[features]
        y_test = test[label]
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        return (x_train, y_train), (x_test, y_test)

    def create_model(self, params):
        return XGBClassifier()


# class OCSVM(AMPMModel):
#
#     def preprocess_data(self, data):
#         train, test = data
#         train = train.fillna(0)
#         test = test.fillna(0)
#         train = train[train['failure'] == 0]
#         features = train.columns[5:]
#         label = train.columns[4]
#         x_train = train[features]
#         y_train = train[label]
#         x_test = test[features]
#         y_test = test[label]
#         pca = PCA(n_components=2)
#         std_scaler = StandardScaler()
#         x_train = pca.fit_transform(std_scaler.fit_transform(x_train))
#         x_test = pca.fit_transform(std_scaler.fit_transform(x_test))
#         return (x_train, y_train), (x_test, y_test)
#
#     def create_model(self, params):
#         return OneClassSVM()


class OCSVM(AMPMModel):

    def preprocess_data(self, data):
        train, test = data
        train = train.fillna(0)
        test = test.fillna(0)
        train = train[train['failure'] == 0]

        # conver to ndarray
        train_arr = np.array(train)
        x_train = train_arr[:, 5:]
        y_train = train_arr[:, 4]
        test_arr = np.array(test)
        x_test = test_arr[:, 5:]
        y_test = test_arr[:, 4]

        # minmaxscaler
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)

        # reshape
        x_train = x_train.reshape(-1, 59, 2, 1)
        x_test = x_test.reshape(-1, 59, 2, 1)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        # encoding
        encoder_input = tf.keras.Input(shape=(59, 2, 1))
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(2)(x)
        encoder_train = tf.keras.Model(encoder_input, encoder_output)
        encoder_train.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=tf.keras.losses.MeanSquaredError())
        encoder_train.fit(x_train, y_train, batch_size=1, epochs=10)
        x_train = encoder_train.predict(x_train)

        encoder_test = tf.keras.Model(encoder_input, encoder_output)
        encoder_test.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=tf.keras.losses.MeanSquaredError())
        encoder_test.fit(x_test, y_test, batch_size=1, epochs=10)
        x_test = encoder_test.predict(x_test)

        return (x_train, y_train), (x_test, y_test)

    def create_model(self, params):
        return OneClassSVM()