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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

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
def train_and_evaluate(df, imputation_method):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    total_num_row=num_of_rows(df)
    n_neighbors=5
    if total_num_row%2:
        n_neighbors=6
    print(f'Número de linhas da base de dados: {total_num_row}\nNúmero de vizinhos usados no treinanmento: {n_neighbors}')
    model = KNeighborsClassifier(n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia com {imputation_method}: {accuracy:.4f}\n\n')
    model_filename = f"{imputation_method}_model.pkl"
    joblib.dump(model, model_filename)
    return model


# Carregar o dataset
data = pd.read_csv('diabetes_dataset.csv')

# Etapa 1: Categorizar as colunas
mostly_null, partially_null, no_null = categorize_columns_by_nulls(data)

# Etapa 2: Remover colunas com valores nulos em sua maioria
data_clean = remove_mostly_null_columns(data, mostly_null)
# Criar e treinar modelos com diferentes métodos de imputação

data_clean=drop_column_by_name(data_clean, 'SkinThickness')
data_clean=drop_column_by_name(data_clean, 'Pregnancies')

partially_null.remove('SkinThickness')
no_null.remove('Pregnancies')

# data_clean = normalizar_minmax(data_clean)

# 1. Imputação com moda
data_mode = data_clean.copy()
print(f'\n\n{data_mode.head()}\n\n')
for column in partially_null:
    data_mode = fill_null_with_mode(data_mode, column)
mode_model=train_and_evaluate(data_mode, 'mode')

# 2. Imputação com mediana
data_median = data_clean.copy()
for column in partially_null:
    data_median = fill_null_with_median(data_median, column)
median_model=train_and_evaluate(data_median, 'median')

# 3. Imputação com média
data_mean = data_clean.copy()
for column in partially_null:
    data_mean = fill_null_with_mean(data_mean, column)
mean_model=train_and_evaluate(data_mean, 'mean')

# 4. Exclusão de linhas com valores nulos
data_dropped = data_clean.copy()
data_dropped = drop_rows_with_nulls(data_dropped)
drop_data_model=train_and_evaluate(data_dropped, 'drop_data')

coluna = colunas(data_clean).copy()
coluna.remove('Outcome')
x_colunas = coluna
print(f'\n\nData frame depois depois do KDD [até a fase 4]:\n\t{data_clean.head()}\n\n')
# # Salvar os conjuntos de dados processados
# data_mode.to_csv('diabetes_mode_imputed.csv', index=False)
# data_median.to_csv('diabetes_median_imputed.csv', index=False)
# data_mean.to_csv('diabetes_mean_imputed.csv', index=False)
# data_dropped.to_csv('diabetes_dropped.csv', index=False)


# Aplicando o modelo e enviando para o servidor
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')

print(f'{data_app.head()}')
mostly_null, partially_null, no_null = categorize_columns_by_nulls(data_app)

X_app = data_app[x_colunas]# Realizar as previsões
# X_app = normalizar_minmax(X_app)

for column in partially_null:
    X_app = fill_null_with_mode(X_app, column)
print(f'{X_app.head()}')

mode_model = joblib.load('mode_model.pkl')
y_pred = mode_model.predict(X_app)

# drop_data_model = joblib.load('drop_data_model.pkl')
# y_pred = drop_data_model.predict(X_app)

# mean_model = joblib.load('mean_model.pkl')
# y_pred = mean_model.predict(X_app)

# median_model = joblib.load('median_model.pkl')
# y_pred = median_model.predict(X_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "VG"

# json para ser enviado para o servidor
data = {'dev_key': DEV_KEY,
        'predictions': pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")