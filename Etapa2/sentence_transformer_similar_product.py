from  sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sklearn
from sklearn.metrics import ndcg_score
from Etapa1.load import load_data

order_products_prior,order_products_train, orders, products = load_data()


def get_similar_products(product_id, top_k=5,embeddings=None):
    product_index = products[products["product_id"] == product_id].index[0]
    product_embedding = embeddings[product_index]
    
    similarities = sklearn.metrics.pairwise.cosine_similarity(
        [product_embedding], embeddings
    )[0]
    
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]

      # Construir resultado
    resultado = products.iloc[similar_indices].copy()
    resultado["similarity"] = similarities[similar_indices]
    return resultado