import pandas as pd

def base_concatenada_comprados_no_comprados(base,base_no_comprados):
    base_comprados=base[["user_id","department_id","product_id","reordered","label"]]
    base_no_comprados=pd.DataFrame({
    'user_id':base_no_comprados["user_id"],
    'department_id':base_no_comprados["department_id"],
    'product_id':base_no_comprados["product_id"],
    'reordered':0,
    'label':base_no_comprados["label"]
    })
    base_final=pd.concat([base_comprados,base_no_comprados])
    return base_final


#user_product_frequency — número de veces que el usuario compró ese producto en su historial

def get_user_product_frequency(base_final):
    frecuencia = (
        base_final.loc[base_final["label"] != 0]
        .groupby(["user_id", "product_id"])
        .size()
    )

    base_final["user_product_frequency"] = (
        base_final.set_index(["user_id", "product_id"])
        .index
        .map(frecuencia)
        .fillna(0)
        .astype("int16")
    )
    return base_final

#user_product_reordered — si el usuario alguna vez reordenó ese producto (1 / 0)


def get_user_product_reordered(base_final):
    base_final.rename({"reordered":"user_product_reordered"}, axis=1, inplace=True)
    return base_final


#product_global_popularity — número total de usuarios distintos que han comprado ese producto

def get_product_global_popularity(base_final):

    frecuencia_usuario = (
        base_final
        .groupby("product_id")
        .agg(product_global_popularity=("user_id", "nunique"))
        .reset_index()
    )


    base_final = base_final.merge(
        frecuencia_usuario,
        on="product_id",
        how="left"
    )
    return base_final


def creacion_features(base_final):
    base_final=get_user_product_frequency(base_final)
    base_final=get_user_product_reordered(base_final)
    base_final=get_product_global_popularity(base_final)
    return base_final
