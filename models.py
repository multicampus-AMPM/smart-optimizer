from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np


def get_matrix(y_test, y_prediction):
    return {
        'accuracy': accuracy_score(y_test, y_prediction),
        'precision': precision_score(y_test, y_prediction, labels=np.unique(y_prediction)),
        'recall': recall_score(y_test, y_prediction, labels=np.unique(y_prediction)),
        'f1': f1_score(y_test, y_prediction, labels=np.unique(y_prediction))
    }


class MODEL:
    def __init__(self, model, params):
        self.model = self.create(model)
        self.name = self.model.__class__.__name__
        self.params = params
        self.scaler = MinMaxScaler()

    def create(self, model):
        raise AttributeError('Model not defined')

    def preprocess(self, X, Y):
        if X is None or X.empty:
            raise ValueError('X must contain values')
        X = X.fillna(0)
        if Y is None:
            # tagging as normal
            Y = [0 for i in range(X.shape[0])]
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)
        return self.encoding(x_train, y_train, x_test, y_test)
    
    def encoding(self, x_train, y_train, x_test, y_test):
        return (x_train, y_train), (x_test, y_test)

    def afterprocess(self, prediction):
        return prediction
    
    def fit(self, x_train, y_train):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, cv=10, refit=True, scoring="recall_micro")
        grid_search.fit(x_train, y_train)
        return grid_search.best_estimator_
    
    def optimize(self, X, Y):
        (x_train, y_train), (x_test, y_test) = self.preprocess(X, Y)
        best_model = self.fit(x_train, y_train)
        prediction = best_model.predict(x_test)
        prediction = self.afterprocess(prediction)
        print(f"{self.name} : {confusion_matrix(y_test, prediction)}")
        return {
            'name': self.name,
            'model': best_model,
            'metrics': get_matrix(y_test, prediction),
            'params': best_model.get_params()
        }


class RF(MODEL):

    def create(self, model):
        if model is None:
            model = RandomForestClassifier()
        return model


class XGB(MODEL):

    def create(self, model):
        if model is None:
            model = XGBClassifier()
        return model


class OCSVM(MODEL):

    def create(self, model):
        if model is None:
            model = OneClassSVM()
        return model

    def encoding(self, x_train, y_train, x_test, y_test):
        # convert
        x_train = np.array(x_train)
        x_test = np.array(x_train)
        y_train = np.array(y_train)
        y_test = np.array(y_train)
        x_train = x_train.reshape(-1, 59, 2, 1)
        x_test = x_test.reshape(-1, 59, 2, 1)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        # left
        left_encoder_input = tf.keras.Input(shape=(59, 2, 1))
        l_x1 = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(left_encoder_input)
        l_x2 = tf.keras.layers.BatchNormalization()(l_x1)
        l_x3 = tf.keras.layers.LeakyReLU()(l_x2)
        l_x4 = tf.keras.layers.Conv2D(64, 3, padding='same')(l_x3)
        l_x5 = tf.keras.layers.BatchNormalization()(l_x4)
        l_x6 = tf.keras.layers.LeakyReLU()(l_x5)
        l_x7 = tf.keras.layers.Flatten()(l_x6)
        left_encoder_output = tf.keras.layers.Dense(1)(l_x7)

        # right
        right_encoder_input = tf.keras.Input(shape=(59, 2, 1))
        r_x1 = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(right_encoder_input)
        r_x2 = tf.keras.layers.BatchNormalization()(r_x1)
        r_x3 = tf.keras.layers.LeakyReLU()(r_x2)
        r_x4 = tf.keras.layers.Conv2D(64, 3, padding='same')(r_x3)
        r_x5 = tf.keras.layers.BatchNormalization()(r_x4)
        r_x6 = tf.keras.layers.LeakyReLU()(r_x5)
        r_x7 = tf.keras.layers.Flatten()(r_x6)
        right_encoder_output = tf.keras.layers.Dense(1)(r_x7)

        # concat
        concat = tf.keras.layers.concatenate([left_encoder_output, right_encoder_output])
        encoder_output = tf.keras.layers.Dense(1)(concat)

        # encoding train
        encoder_train = tf.keras.Model(inputs=[left_encoder_input, right_encoder_input], outputs=encoder_output)
        encoder_train.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=tf.keras.losses.MeanSquaredError())
        encoder_train.fit(x_train, y_train, batch_size=1, epochs=10)
        x_train = encoder_train.predict(x_train)
        
        # test
        encoder_input = tf.keras.Input(shape=(59, 2, 1))
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(1)(x)

        # encoding test
        encoder_test = tf.keras.Model(encoder_input, encoder_output)
        encoder_test.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss=tf.keras.losses.MeanSquaredError())
        encoder_test.fit(x_test, y_test, batch_size=1, epochs=1)
        x_test = encoder_test.predict(x_test)
        return (x_train, y_train), (x_test, y_test)
    
    def afterprocess(self, prediction):
        prediction = np.where(prediction == 1, 0, prediction)
        prediction = np.where(prediction == -1, 1, prediction)
        return prediction
    
    def fit(self, x_train, y_train):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, cv=10, refit=True, scoring="recall_micro")
        grid_search.fit(x_train, np.array([0 for i in range(x_train.shape[0])]))
        return grid_search.best_estimator_