from flask import Flask
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client.core import GaugeMetricFamily
import os
import mlflow
from models import RF, XGB, OCSVM
from register import log_model
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


rf = 'RandomForestClassifier'
xgb = 'XGBClassifier'
ocsvm = 'OneClassSVM'

MODELS = {
    rf: RF,
    xgb: XGB,
    ocsvm: OCSVM
}

PARAMS = {
    rf: {
        'n_estimators': [10, 50, 100],
        'max_features': ['auto', 3, 5]
    },
    xgb: {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [6]
    },
    ocsvm: {
        # kernel='precomputed'can only be used when passing a (n_samples, n_samples) data matrix that represents pairwise similarities for the samples instead of the traditional (n_samples, n_features) rectangular data matrix.
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'nu': [0.1, 0.3, 0.5, 0.7]
    }
}


FEATURES = ['read-error-rate-normal',
            'read-error-rate-raw',
            'throughput-performance-normal',
            'throughput-performance-raw',
            'spin-up-time-normal',
            'spin-up-time-raw',
            'start/stop-count-normal',
            'start/stop-count-raw',
            'reallocated-sectors-count-normal',
            'reallocated-sectors-count-raw',
            'seek-error-rate-normal',
            'seek-error-rate-raw',
            'seek-time-performance-normal',
            'seek-time-performance-raw',
            'power-on-hours-normal',
            'power-on-hours-raw',
            'spin-retry-count-normal',
            'spin-retry-count-raw',
            'recalibration-retries-normal',
            'recalibration-retries-raw',
            'power-cycle-count-normal',
            'power-cycle-count-raw',
            'soft-read-error-rate-normal',
            'soft-read-error-rate-raw',
            'current-helium-level-normal',
            'current-helium-level-raw',
            'available-reserved-space-normal',
            'available-reserved-space-raw',
            'ssd-wear-leveling-count-normal',
            'ssd-wear-leveling-count-raw',
            'unexpected-power-loss-count-normal',
            'unexpected-power-loss-count-raw',
            'power-loss-protection-failure-normal',
            'power-loss-protection-failure-raw',
            'wear-range-delta-normal',
            'wear-range-delta-raw',
            'used-reserved-block-count-total-normal',
            'used-reserved-block-count-total-raw',
            'unused-reserved-block-count-total-normal',
            'unused-reserved-block-count-total-raw',
            'program-fail-count-total-normal',
            'program-fail-count-total-raw',
            'erase-fail-count-normal',
            'erase-fail-count-raw',
            'sata-downshift-error-count-normal',
            'sata-downshift-error-count-raw',
            'end-to-end-error-normal',
            'end-to-end-error-raw',
            'reported-uncorrectable-errors-normal',
            'reported-uncorrectable-errors-raw',
            'command-timeout-normal',
            'command-timeout-raw',
            'high-fly-writes-normal',
            'high-fly-writes-raw',
            'temperature-difference-normal',
            'temperature-difference-raw',
            'g-sense-error-rate-normal',
            'g-sense-error-rate-raw',
            'power-off-retract-count-normal',
            'power-off-retract-count-raw',
            'load-cycle-count-normal',
            'load-cycle-count-raw',
            'temperature-normal',
            'temperature-raw',
            'hardware-ecc-recovered-normal',
            'hardware-ecc-recovered-raw',
            'reallocation-event-count-normal',
            'reallocation-event-count-raw',
            'current-pending-sector-count-normal',
            'current-pending-sector-count-raw',
            '(offline)-uncorrectable-sector-count-normal',
            '(offline)-uncorrectable-sector-count-raw',
            'ultradma-crc-error-count-normal',
            'ultradma-crc-error-count-raw',
            'multi-zone-error-rate-normal',
            'multi-zone-error-rate-raw',
            'data-address-mark-errors-normal',
            'data-address-mark-errors-raw',
            'flying-height-normal',
            'flying-height-raw',
            'vibration-during-write-normal',
            'vibration-during-write-raw',
            'disk-shift-normal',
            'disk-shift-raw',
            'loaded-hours-normal',
            'loaded-hours-raw',
            'load/unload-retry-count-normal',
            'load/unload-retry-count-raw',
            'load-friction-normal',
            'load-friction-raw',
            'load/unload-cycle-count-normal',
            'load/unload-cycle-count-raw',
            "load-'in'-time-normal",
            "load-'in'-time-raw",
            'life-left-(ssds)-normal',
            'life-left-(ssds)-raw',
            'endurance-remaining-normal',
            'endurance-remaining-raw',
            'media-wearout-indicator-(ssds)-normal',
            'media-wearout-indicator-(ssds)-raw',
            'average-erase-count-and-maximum-erase-count-normal',
            'average-erase-count-and-maximum-erase-count-raw',
            'good-block-count-and-system(free)-block-count-normal',
            'good-block-count-and-system(free)-block-count-raw',
            'head-flying-hours-normal',
            'head-flying-hours-raw',
            'total-lbas-written-normal',
            'total-lbas-written-raw',
            'total-lbas-read-normal',
            'total-lbas-read-raw',
            'read-error-retry-rate-normal',
            'read-error-retry-rate-raw',
            'minimum-spares-remaining-normal',
            'minimum-spares-remaining-raw',
            'newly-added-bad-flash-block-normal',
            'newly-added-bad-flash-block-raw',
            'free-fall-protection-normal',
            'free-fall-protection-raw']


class OptimizerExporter(object):

    def __init__(self, logger):
        self.logger = logger
        self.url = os.environ['prom']
        self.queries = ['collectd_smart_smart_attribute_current[1h]', 'collectd_smart_smart_attribute_pretty[1h]']
        self.models = self.load_model()
        self.logger.error(self.models)

    
    def load_model(self):
        model_list = list()
        for model_name, model_obj in MODELS.items():
            try:
                ref = f'models:/{model_name}/Production'
                model = mlflow.sklearn.load_model(ref) if model_name == rf else mlflow.pyfunc.load_model(ref)
                model_list.append(model_obj(model, PARAMS[model_name]))
            except Exception:
                continue
        return model_list
    
    def from_prometheus(self):
        result = list()
        try:
            for query in self.queries:
                response = requests.get(url=self.url, params={'query': query}, timeout=10)
                result.extend(response.json()['data']['result'])
            if not result:
                raise ValueError('No result from prometheus')
        except Exception as e:
            self.logger.error(e)
            return None

        servers = dict()
        for metric in result:
            server_name = metric['metric']['instance']
            metrics = servers.get(server_name)
            if metrics is None:
                servers[server_name] = dict()
                metrics = servers[server_name]

            attr_name = metric['metric']['type']
            attr_name += '-raw' if metric['metric']['__name__'].endswith('pretty') else '-normal'
            rows = metrics.get(attr_name)
            if rows is None:
                metrics[attr_name] = list()
                rows = metrics[attr_name]

            values = metric['values']
            for value in values:
                rows.append(value[1])
        return servers

    def add_features(self, metrics):
        first_key = next(iter(metrics))
        length = len(metrics[first_key])
        for feature in FEATURES:
            from_metrics = metrics.get(feature)
            if from_metrics is None:
                metrics[feature] = [0 for i in range(length)]
        new_metrics = dict()
        for f in FEATURES:
            new_metrics[f] = metrics[f]
        return new_metrics

    def collect(self):
        gmt = GaugeMetricFamily(name="ampm_model_optimization", documentation='Result of Model Optimization', labels=['run', 'name', 'metric'])    
        data = self.from_prometheus()
        if data:
            for server_name, metrics in data.items():
                dataset = pd.DataFrame(self.add_features(metrics))
                if dataset.empty or dataset.shape[0] < 5:
                    continue
                runs = list()
                try:
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        tasks = [executor.submit(model.optimize, dataset, None) for model in self.models]
                        for task in as_completed(tasks):
                            runs.append(task.result())
                except Exception as e:
                    self.logger.error(e)
                    continue

                for result in runs:
                    with mlflow.start_run() as run:
                        mlflow.set_tags({'estimator_name': result['name']})
                        mlflow.log_metrics(result['metrics'])
                        mlflow.log_params(result['params'])
                        if result['name'] == xgb:
                            mlflow.xgboost.log_model(result['model'], 'model')
                        else:
                            mlflow.sklearn.log_model(result['model'], 'model')
                    self.add_metric(run.info.run_id, result['name'], result['metrics'], gmt)
                # TODO 모델 갈아끼기
        yield gmt
    
    def add_metric(self, run_id, model_name, run_metrics, gmt):
        for metric_name, value in run_metrics.items():
            gmt.add_metric([run_id, model_name, metric_name], value)


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
    log_model()
    exporter.registry.register(OptimizerExporter(app.logger))
    app.run(host=os.environ.get('host'), port=os.environ.get('port'))
