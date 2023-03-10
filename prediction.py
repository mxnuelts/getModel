import os
import pandas as pd
import numpy as np
import sys
pd.set_option("display.float_format", lambda x: "%.3f" % x)
from datetime import datetime
import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from metrics import Auc, LogLoss, RecallAtPrecision, PrecisionAtRecall, RecallAtFpr

import warnings

warnings.filterwarnings("ignore")


class PrediccionModelo(object):
    def __init__(
        self,
        categorical_features,
        features_to_drop,
        method="",
        path_save="outputs",
    ):
        self.categorical_features = categorical_features
        self.features_to_drop = features_to_drop
        self.method = method
        self.path_save = path_save

    def load_data(self, table):
        df = pd.read_sql_query(table)
        print(f"Dataframe shape: {df.shape}")

        return df

    def data_prepro(self, data):
        columnas = [x for x in data.columns if x not in self.features_to_drop]
        id=data['id'] 
        X_test = data[columnas]        
        print(f"Data shape: {X_test.shape}")

        if self.method == "scale":
            scaled = StandardScaler()
            X_test = pd.DataFrame(scaled.transform(X_test), columns=X_test.columns)
        elif self.method == 'minmax':
            scaled = MinMaxScaler()
            X_test = pd.DataFrame(scaled.transform(X_test), columns=X_test.columns)

        return X_test,id

    def predict(self, model_file, X,nro_documento_ls):
        model = lgb.Booster(model_file=model_file)
        print("Predicting model")
        now = datetime.now()
        X['score'] = model.predict(X)
        doc = pd.DataFrame({"id": id})
        res_filename = f"preds/pred_full_v5.csv"
        res = pd.concat([doc.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        res.to_csv(os.path.join(self.path_save, res_filename),sep=';')


if __name__ == "__main__":

    categorical_features = [
        "Sexo_Masculino", 
        "Tipo de hogar_Departamento",
        "Estado civil_Divorciado",
        "Estado civil_Soltero",
        "Estado civil_Viudo",
        "Tipo de trabajo_Empresario",
        "Tipo de trabajo_Independiente",
        "Educación_Preparatoria",
        "Educación_Universidad",
        "Tipo de seguro_Hogar",
        "Tipo de seguro_Salud",
        "Tipo de seguro_Vida"
    ]
    features_to_drop = [
        'ID del cliente',
        'start_date',
        'target'
    ]

    eval_metrics = [
        Auc(metric_name="auc"),
        LogLoss(metric_name="log_loss"),
        RecallAtPrecision(precision=0.9, metric_name="recall_at_precision"),
        PrecisionAtRecall(recall=0.85, metric_name="precision_at_recall"),
        RecallAtFpr(fpr=0.1, metric_name="recall_at_fpr"),
    ]

    model = PrediccionModelo(
        categorical_features=categorical_features,
        features_to_drop=features_to_drop,
        eval_metrics=eval_metrics,
        #method="minmax",
    )
    print("Model instantiated!")
    df = model.load_data(table="data/out/data_proc.csv")
    
    data_vars,id = model.data_prepro(df)
    
    # command to predict from saved model
    model_file = "outputs\models\model_09-03-2023_23_34_58.txt"
    model.predict(model_file, data_vars,id)