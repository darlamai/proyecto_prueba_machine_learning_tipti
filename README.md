# proyecto_prueba_machine_learning_tipti
Proyecto de prueba para ingresar a Tipti (Machine Learning Engineer)

# Sistema de Recomendación de Productos 

## Descripción

Este proyecto implementa un sistema de recomendación de productos utilizando el dataset de https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset.

El desarrollo se divide en tres etapas: un modelo de ranking, un sistema de embeddings para productos y un agente conversacional capaz de responder consultas sobre recomendaciones.

---

## Estructura del proyecto

```
├── Data/
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   └── products.csv
│
├── Etapa1
|  ├── data_limpia_features.py
|  ├── load.py
|  ├── merge.py
|  ├── model.py
|  ├── no_comprados.py

├── Etapa2
|  ├── sentence_transformer_similar_product.py

├── Etapa3
|  ├── .env
|  ├── langchainscript.py

└── Prueba.ipynb


```

---

# Etapa 1 - Modelo de Ranking

Se construyó un sistema de recomendación basado en **Learning to Rank** utilizando **XGBoost Ranker**.

### Proceso

* Integración de las tablas del dataset.
* Construcción de candidatos positivos y negativos ( se agrupò por usuario y departamento y la misma cantidad de positivos se muestreó para los negativos).
* Entrenamiento mediante `XGBRanker`.
* Evaluación mediante **NDCG@10**  sobre el dataset de train.

### Features utilizadas

* user_product_frequency
* user_product_reordered
* product_popularity_scaled

---

# Etapa 2 - Embeddings

Se generaron embeddings semánticos de productos utilizando:

* Sentence Transformers
* Modelo: `all-MiniLM-L6-v2`

Funcionalidades implementadas:

* búsqueda de productos similares mediante la similitud mediante Cosine Similarity

---

# Etapa 3 - Agente Conversacional

Se implementó un agente utilizando:

* LangChain
* LangGraph

Herramientas disponibles:

* `predict_reorder(user_id, product_id)`
* `get_similar_products(product_id)`
* `get_user_history(user_id)`

El agente responde preguntas relacionadas con recomendaciones y comportamiento histórico de compra de los usuarios.
El key de la api de openai se almacena en un archivo .env 

---

---

## Ejecución

1. Colocar los archivos CSV dentro de la carpeta `Data/` y cargar librerías del archivo requirements.txt.
2. Ejecutar el notebook principal.
3. Entrenar el modelo.
4. Consultar el agente mediante las herramientas implementadas.

---

## Tecnologías

* Python
* Pandas
* NumPy
* XGBoost
* Scikit-Learn
* Sentence Transformers
* LangChain
* LangGraph
