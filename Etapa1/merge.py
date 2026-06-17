import pandas as pd


def merge_tablas (prior_train,orders, products,prior_train_test):
    orders_train_test=orders[orders['eval_set']==prior_train_test]
    base1=pd.merge(prior_train,orders_train_test, on='order_id',how="left")
    base=pd.merge(base1,products, on='product_id',how="left")
    return base


def labeling(base):
    base["label"]=[2 if x==1 else (1 if x==0 else None) for x in base["reordered"]]
    return base



