#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests
import matplotlib.pyplot as plt
import seaborn as sns

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('/Users/davideneas/Library/CloudStorage/OneDrive-InstitutodeComputação-UniversidadeFederaldeAlagoas/4. Quarto Periodo/PM/atividades/mlclass/01_Preprocessing/diabetes_dataset.csv')

# Função para categorizar as colunas com base na quantidade de valores nulos
def categorize_columns_by_nulls(df):
    column_lst = list(df.columns)
    mostly_null = []
    partially_null = []
    no_null = []

    total_rows = len(df)  # Obtém o número total de linhas

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

# Função para preencher valores nulos com zero
def fill_null_with_zero(df, partially_null):
    for column in partially_null:
        df[column] = df[column].fillna(0)
    return df

# Função para remover coluna por nome
def drop_column_by_name(df, column='id'):
    if column in df.columns:
        return df.drop(column, axis=1)
    print(f"Column '{column}' not found in the DataFrame.")
    return df

# Etapa 1: Categorizar as colunas
print(' - Categorizando as colunas')
mostly_null, partially_null, no_null = categorize_columns_by_nulls(data)

# Etapa 2: Remover colunas com valores nulos em sua maioria
print(' - Removendo colunas com valores nulos em sua maioria')
data = remove_mostly_null_columns(data, mostly_null)

# Etapa 3: Preencher valores nulos parciais com zero
print(' - Preenchendo valores nulos parciais com zero')
data = fill_null_with_zero(data, partially_null)

# Etapa 4: Inspecionar as colunas para verificar se ainda há valores nulos
print(' - Verificando valores nulos na base de dados')
mostly_null, partially_null, no_null = categorize_columns_by_nulls(data)

# Selecione as colunas de X e y
X = data.drop(columns=['Outcome']) # Remover a coluna 'Outcome' que é o alvo
y = data['Outcome']  # A coluna 'Outcome' é o alvo

# Verificar se há valores ausentes antes de treinar o modelo
if X.isnull().sum().sum() > 0:
    print("Existem valores ausentes em X. Corrija antes de treinar o modelo.")
    exit()
else:
    print("Não há valores ausentes em X. Continuando com o treino do modelo.")

# Criando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Aplicando o modelo e enviando para o servidor
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('/Users/davideneas/Library/CloudStorage/OneDrive-InstitutodeComputação-UniversidadeFederaldeAlagoas/4. Quarto Periodo/PM/atividades/mlclass/01_Preprocessing/diabetes_app.csv')

# Garantir que as colunas de X estejam no data_app
data_app = data_app[X.columns]  # Ajuste para que as colunas em data_app correspondam às de X

y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Eneas"

# json para ser enviado para o servidor
data = {'dev_key': DEV_KEY,
        'predictions': pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")