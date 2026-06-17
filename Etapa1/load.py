import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRanker
import sklearn
import os



def load_data():

    data_path = "Data"

    order_products_prior = pd.read_csv(
        os.path.join(data_path, "order_products__prior.csv")
    )

    order_products_train = pd.read_csv(
        os.path.join(data_path, "order_products__train.csv")
    )

    orders = pd.read_csv(
        os.path.join(data_path, "orders.csv")
    )

    products = pd.read_csv(
        os.path.join(data_path, "products.csv")
    )

    return order_products_prior, order_products_train, orders, products









