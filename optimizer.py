import os
import mlflow.sklearn


def save_model():
    runs = mlflow.search_runs(order_by=['metrics.training_score'], max_results=1)
    mlflow.sklearn.save_model(runs.loc[0], os.environ['repo'])


def parse_env():
    # use get function instead of key access to avoid error
    repo = os.environ.get('repo')

    # TODO regex validation
    if repo is None:
        os.environ['repo'] = '/tmp/smart'


if __name__ == '__main__':
    parse_env()
    save_model()