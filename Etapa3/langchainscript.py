import os
import pandas as pd
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from sentence_transformer_similar_product import *
from merge import *
from data_limpia_features import *
from model import *
from load import load_data
from no_comprados import *
from  sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
from dotenv import load_dotenv  # <-- Para leer tu archivo .env



from typing import List, Dict, Any

from load import load_data

order_products_prior,order_products_train, orders, products = load_data()

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(products["product_name"].tolist(),show_progress_bar=True)

# Cargar las variables de entorno del archivo .env antes de cualquier otra cosa
load_dotenv() 

###funciones

def get_user_history(order_products_prior,orders, products,user_id: int) -> List[Dict[str, Any]]:
    """
    Retorna la lista de productos más comprados (historial) de un usuario.
    Si el usuario no tiene historial o es nuevo, retorna una lista vacía.
    """
    orders_train_test=orders[orders['eval_set']=='prior']
    base_prior=pd.merge(order_products_prior,orders_train_test, on='order_id',how="left")
    ###filtramos el usuario
    usuario_base_prior=base_prior[base_prior["user_id"]==user_id]
    tabla_conteo=usuario_base_prior.groupby("product_id").size().reset_index(name="conteo")
    historial = tabla_conteo.merge(
    products[["product_id", "product_name"]],
    on="product_id",
    how="left")
    productos_mas_comprados = historial.sort_values("conteo", ascending=False)["product_name"].tolist()[:5]

    return {user_id: productos_mas_comprados}


def get_similar_products_langchain(
    product_id: int,
    k: int = 2
) -> List[Dict[str, Any]]:
    """
    Retorna los k productos más similares a un producto dado.
    """

    try:
        lista_similares = get_similar_products(product_id, top_k=k,embeddings=embeddings)

        return (
            lista_similares[
                ["product_id", "product_name", "similarity"]
            ]
            .to_dict(orient="records")
        )

    except (IndexError, KeyError):
        return []
    


def predict_reorder(user_id: str, product_id: str) -> float:
    """
    Llama al clasificador predictivo (Etapa 1). Calcula la probabilidad (score de 0 a 1) 
    de que el usuario vuelva a pedir (reorder) ese producto específico.
    """
    base=merge_tablas(order_products_prior, orders, products,"prior")
    base=labeling(base)
    dic_base,dic_products,dic_diferencia=calculo_diferentes(products,base)
    base_no_comprados=base_final_no_comprados(dic_diferencia)
    base_final=base_concatenada_comprados_no_comprados(base,base_no_comprados)
    base_final=creacion_features(base_final)
    X,y,base_final_model=variables_modelo(base_final)
    ranker,group=modelo_grupo(base_final_model)
    ranker=entrenar_modelo(ranker,X,y,group)
    base_final["score"] = ranker.predict(X)

    resultado = base_final.loc[
        (base_final["user_id"] == user_id)
        & (base_final["product_id"] == product_id),
        ["user_id", "product_id", "score"]
    ]

    if resultado.empty:
        return {
            "user_id": user_id,
            "product_id": product_id,
            "score": None,
            "message": "No se encontró el usuario o el producto."
        }

    return resultado.iloc[0].to_dict()




######herramientas


@tool
def tool_get_user_history(user_id: int) -> List[str]:
    """
    Retorna la lista de los nombres de productos más comprados (historial) de un usuario.
    Úsala al inicio del flujo usando el ID numérico del usuario.
    Si retorna una lista vacía, significa que el usuario es nuevo.
    """
    # Pasamos las variables de entorno/globales (order_products_prior, orders, products) 
    # que ya tienes cargadas en tu memoria de ejecución.
    resultado_dict = get_user_history(order_products_prior, orders, products, user_id=user_id)
    
    # Tu función devuelve un dict {user_id: [lista_productos]}. 
    # Extraemos solo la lista de strings para que el LLM la procese de forma más limpia.
    return resultado_dict.get(user_id, [])



@tool
def tool_get_similar_products(product_id: int, k: int = 2) -> List[Dict[str, Any]]:
    """
    Llama al modelo de embeddings vectoriales. Dado el ID numérico de un producto, 
    busca los 'k' productos alternativos más similares en el catálogo de la tienda.
    Retorna una lista de diccionarios con 'product_id', 'product_name' y 'similarity'.
    """
    # Ejecuta directamente tu función puente que maneja los embeddings globales
    return get_similar_products_langchain(product_id=product_id, k=k)



@tool
def tool_predict_reorder(user_id: int, product_id: int) -> float:
    """
    Calcula la probabilidad predictiva (score de 0.0 a 1.0) de que el usuario vuelva a pedir 
    ese producto específico. Úsala para evaluar la relevancia final de las recomendaciones sugeridas.
    """
    # Forzamos conversión a int si el LLM envía strings, para evitar errores de tipo con Pandas
    u_id = int(user_id)
    p_id = int(product_id)
    
    # Nota de optimización académica: Si tu función predict_reorder entrena el modelo 
    # desde cero en cada llamada (modelo_grupo, entrenar_modelo), esta llamada puede demorar varios segundos.
    resultado_dict = predict_reorder(str(u_id), str(p_id))
    
    # Extraemos el score numérico final. Si no existe o da error, enviamos un score neutro (0.50)
    score = resultado_dict.get("score")
    if score is None or pd.isna(score):
        return 0.50
    return float(score)


# Agrupamos todas las herramientas necesarias para la orquestación
tools = [
    tool_get_user_history, 
    tool_get_similar_products, 
    tool_predict_reorder
]







