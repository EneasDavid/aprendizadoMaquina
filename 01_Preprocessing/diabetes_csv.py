#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import requests
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def colunas(df):
    return list(df.columns)

def num_of_rows(df):
    return len(df)
# Função para categorizar as colunas com base na quantidade de valores nulos
def categorize_columns_by_nulls(df):
    column_lst = colunas(df)
    mostly_null = []
    partially_null = []
    no_null = []

    total_rows = num_of_rows(df)  # Obtém o número total de linhas

    for column in column_lst:
        missing_count = df[column].isnull().sum()
        percentage = (missing_count * 100) / total_rows

        if percentage >= 60:
            mostly_null.append(column)
        elif missing_count > 0:
            partially_null.append(column)
        else:
            no_null.append(column)

        print(f'Feature Name: {column}')
        print(f'Number of missing values: {missing_count} out of {total_rows}')
        print(f'Missing percentage: {percentage:.2f}%')
        print()
    if len(mostly_null) + len(partially_null) + len(no_null) == len(column_lst):
        print("All columns categorized successfully.")
    else:
        print("Error: Some columns were not categorized.")

    return mostly_null, partially_null, no_null

# Função para remover colunas com muitos valores nulos
def remove_mostly_null_columns(df, mostly_null):
    return df.drop(mostly_null, axis=1)

# Funções para preencher valores nulos
def fill_null_with_mode(df, column):
    mode_value = df[column].mode()[0]
    print(f'Moda de {column}: {mode_value}')
    df[column] = df[column].fillna(mode_value)
    return df

def fill_null_with_median(df, column):
    median_value = df[column].median()
    print(f'Mediana de {column}: {median_value}')
    df[column] = df[column].fillna(median_value)
    return df

def fill_null_with_mean(df, column):
    mean_value = df[column].mean()
    print(f'Média de {column}: {mean_value}')
    df[column] = df[column].fillna(mean_value)
    return df

def drop_rows_with_nulls(df):
    return df.dropna()

# Função para remover coluna por nome
def drop_column_by_name(df, column):
    if column in colunas(df):
        return df.drop(column, axis=1)
    print(f"Column '{column}' not found in the DataFrame.")
    return df

# Função para normalizar os valores da coluna
def normalizar_minmax(df):
    return (df - df.min()) / (df.max() - df.min())

# Função para treinar e avaliar o modelo
def identificar_zeros(df):
    cols_with_zeros = [col for col in df.columns if (df[col] == 0).any()]
    return cols_with_zeros



print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

mostly_null, partially_null, no_null = categorize_columns_by_nulls(data)

# # 1. Imputação com moda
# for column in partially_null:
#     data = fill_null_with_mode(data, column)

# # 2. Imputação com mediana
for column in partially_null:
    data = fill_null_with_median(data, column)

# 3. Imputação com média
# for column in partially_null:
#     data = fill_null_with_mean(data, column)

# 4. Exclusão de linhas com valores nulos
# data = drop_rows_with_nulls(data)

# 5. Remoção de colunas que não dá pra preencher
data = remove_mostly_null_columns(data, mostly_null)

colunas_zero=identificar_zeros(data)
print(f'\n\nColunas com valores zerados: {colunas_zero}\n\n')


# Criando X and y par ao algorítmo de aprendizagem de máquina.
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness'
                , 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]

y = data.Outcome

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]

y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "VG"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")