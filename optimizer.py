from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client.core import GaugeMetricFamily
import os
import mlflow
from models import RF, XGB, OCSVM, register_model
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PARAMS = {
    'RF': {
        'n_estimators': [10, 50, 100],
        'max_features': ['auto', 3, 5]
    },
    'XGB': {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [6]
    },
    'OCSVM': {
        # kernel='precomputed'can only be used when passing a (n_samples, n_samples) data matrix that represents pairwise similarities for the samples instead of the traditional (n_samples, n_features) rectangular data matrix.
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'nu': [0.1, 0.3, 0.5, 0.7]
    }
}


class SmartOptimizerExporter(object):

    def __init__(self, prometheus_url, logger, train, test):
        self.url = prometheus_url
        self.logger = logger
        self.train = train
        self.test = test

    def load_data(self):
        if not os.path.exists(self.train):
            raise FileNotFoundError('train file not found')
        if not os.path.exists(self.test):
            raise FileNotFoundError('test file not found')
        return pd.read_csv(self.train), pd.read_csv(self.test)

    def collect(self):
        data = self.load_data()
        models = [RF(data, PARAMS['RF']), XGB(data, PARAMS['XGB']), OCSVM(data, PARAMS['OCSVM'])]
        runs = list()
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [executor.submit(model.optimize) for model in models]
            for task in as_completed(tasks):
                runs.append(task.result())

        self.logger.error(runs)
        for run in runs:
            with mlflow.start_run():
                mlflow.set_tags({'estimator_name': run['name']})
                mlflow.log_metrics(run['metrics'])
                mlflow.log_params(run['params'])
                if run['name'] == 'XGBClassifier':
                    mlflow.xgboost.log_model(run['model'], 'model')
                else:
                    mlflow.sklearn.log_model(run['model'], 'model')
            gmt = GaugeMetricFamily(name=f"{run['name'].lower()}_hyperparameter", documentation='', labels=['name'])
            for key, value in run['params'].items():
                gmt.add_metric(key, value)
            yield gmt


app = Flask(__name__)
exporter = PrometheusMetrics(app)


@app.route('/favicon.ico')
@exporter.do_not_track()
def favicon():
    return 'ok'


@app.route('/')
@exporter.do_not_track()
def main():
    """ context root """
    return """
        <html>
            <head><title>Optimizer Exporter</title></head>
            <body>
                <h1>Optimizer Exporter</h1>
                <p><a href='/metrics'>Metrics</a></p>
            </body>
        </html>
    """


def parse_env():
    # use get function instead of key access to avoid error
    host = os.environ.get('host')
    port = os.environ.get('port')
    repo = os.environ.get('repo')
    prom = os.environ.get('prom')

    # TODO regex validation
    if host is None:
        os.environ['host'] = '0.0.0.0'
    if port is None:
        os.environ['port'] = '9109'
    if repo is None:
        os.environ['repo'] = 'model-repository'
    if prom is None:
        os.environ['prom'] = "prometheus:9090"
    os.environ['prom'] = f"http://{os.environ['prom']}/api/v1/query"
    os.environ['tracking'] = f'http://{os.environ["repo"]}:5000'
    os.environ['ftp'] = f'ftp://mlflow:mlflow@{os.environ["repo"]}/mlflow/artifacts/'

if __name__ == '__main__':
    parse_env()
    register_model()
    # mlflow.set_tracking_uri(os.environ['repo'])
    # exporter.registry.register(SmartOptimizerExporter(os.environ['prom'], app.logger, 'data/smart_train.csv', 'data/cd_fail_col.csv'))
    app.run(host=os.environ.get('host'), port=os.environ.get('port'))
