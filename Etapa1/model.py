import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRanker
import sklearn
from sklearn.metrics import ndcg_score

def variables_modelo (base_final):
 
    base_final = base_final.sort_values(by="user_id").reset_index(drop=True)

    #Normalizar popularidad por grupo de usuario
    base_final['pop_max_user'] = base_final.groupby('user_id')['product_global_popularity'].transform('max')
    base_final['product_popularity_scaled'] = base_final['product_global_popularity'] / (base_final['pop_max_user'] + 1e-5)
    base_final_model=base_final

        # Separate features (X) and target relevance labels (y)
    X = base_final_model[[ "product_id", "user_product_frequency", "user_product_reordered", "product_popularity_scaled"]]
    y = base_final_model["label"]
    return X,y,base_final_model


def modelo_grupo(base_final_model):
    ranker = XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.1,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    base_final_model = (
    base_final_model
    .sort_values("user_id")
    .reset_index(drop=True))
        
    group = (
        base_final_model
        .groupby("user_id")
        .size()
        .tolist()
    )
    return ranker,group


def entrenar_modelo(ranker, X, y, group):

    ranker.fit(
        X,
        y,
        group=group
    )

    return ranker


def  create_ranking(base_final,score):
    ranking = (
    base_final
    .sort_values(
        ["user_id", score],
        ascending=[True,False]
    )
    )
    return ranking




def metricas_modelo(ranking,score):
    ndcg = []

    for _, grupo in ranking.groupby("user_id"):

        if len(grupo) < 2:
            continue

        y_true = grupo["label"].values
        y_score = grupo[score].values

        ndcg.append(
            ndcg_score(
                [y_true],
                [y_score],
                k=10
            )
        )

    ndcg10 = sum(ndcg) / len(ndcg)

    return ndcg10



    