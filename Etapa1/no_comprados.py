import pandas as pd

dic_products={}
dic_base={}
dic_diferencia = {}

def calculo_diferentes (products,base):

    dic_products = products.groupby("department_id")["product_id"].apply(lambda x: list(set(x))).to_dict()

    aux = (
        base.groupby(["user_id", "department_id"])["product_id"]
            .apply(lambda x: list(set(x)))
    )

    dic_base = {
        (user, dept, len(products)): products
        for (user, dept), products in aux.items()
    }

    for (user, dept, n), productos_usuario in dic_base.items():

        comprados = set(productos_usuario)

        diferencia = []

        for producto in dic_products[dept]:

            if producto not in comprados:
                diferencia.append(producto)

                if len(diferencia) == n:
                    break

        dic_diferencia[(user, dept, n)] = diferencia

    return dic_base,dic_products,dic_diferencia


def base_final_no_comprados(dic_diferencia):
    rows = []
    for (user_id, department_id, count_product_id), products in dic_diferencia.items():
        for pid in products:
            rows.append({
                "user_id": user_id,
                "department_id": department_id,
                "product_id": pid
            })

    base_no_comprados = pd.DataFrame(rows)
    base_no_comprados["label"]=0
    return base_no_comprados

### de prueba de los comprados

def conteo_comprados_user_dep(base):
    conteo_comprados_dep1=base.groupby(["user_id","department_id"]).agg({"product_id":"count"}).reset_index()
    return conteo_comprados_dep1