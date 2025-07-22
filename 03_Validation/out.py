import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import requests

# =====================================
# Função para remover outliers com IQR
# =====================================
def remover_outliers_iqr(df):
    # Aqui, estamos pegando todas as colunas do dataframe, exceto 'type'
    colunas = [col for col in df.columns if col != "type"]
    
    # Calculando o primeiro quartil (Q1) e o terceiro quartil (Q3) para cada coluna
    Q1 = df[colunas].quantile(0.25)
    Q3 = df[colunas].quantile(0.75)
    
    # A diferença entre Q3 e Q1 é o IQR (Intervalo Interquartil)
    IQR = Q3 - Q1

    # Remover linhas com valores fora do intervalo permitido (outliers)
    df_limpo = df[~((df[colunas] < (Q1 - 1.5 * IQR)) | (df[colunas] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Plota os boxplots para mostrar os outliers antes e depois da remoção
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    sns.boxplot(data=df[colunas], ax=axes[0])  # Antes
    axes[0].set_title('Antes da remoção de outliers')
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(data=df_limpo[colunas], ax=axes[1])  # Depois
    axes[1].set_title('Após a remoção de outliers')
    axes[1].tick_params(axis='x', rotation=45)

    fig.tight_layout()
    plt.show()

    # Retorna o dataframe sem os outliers
    return df_limpo

# =====================================
# Função para tratar correlação
# =====================================
def remover_colunas_correlacionadas(data, threshold=0.95):
    # Seleciona as colunas numéricas (sem contar a coluna 'type')
    numerical_features = [col for col in data.select_dtypes(include=[np.number]).columns if col != "type"]
    
    # Calcula a matriz de correlação entre as colunas numéricas
    corr_matrix_before = data[numerical_features].corr()

    # Identifica colunas altamente correlacionadas (maior que o limite especificado)
    upper = corr_matrix_before.where(np.triu(np.ones(corr_matrix_before.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]

    # Mostra quais colunas serão removidas
    print("Removendo colunas altamente correlacionadas:", to_drop)
    colunas_antes = len(data.columns)
    
    # Remove as colunas altamente correlacionadas
    data = data.drop(columns=to_drop)
    colunas_depois = len(data.columns)

    print(f"Quantidade de colunas antes: {colunas_antes}, depois: {colunas_depois}")

    # Recalcular matriz de correlação após remoção
    numerical_features = [col for col in data.select_dtypes(include=[np.number]).columns if col != "type"]
    corr_matrix_after = data[numerical_features].corr()
    
    # Plota os mapas de calor antes e depois da remoção das colunas correlacionadas
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    sns.heatmap(corr_matrix_before, annot=True, cmap="coolwarm", fmt='.2f', ax=axes[0])
    axes[0].set_title("Correlação - Antes", fontsize=14)
    
    sns.heatmap(corr_matrix_after, annot=True, cmap="coolwarm", fmt='.2f', ax=axes[1])
    axes[1].set_title("Correlação - Depois", fontsize=14)
    plt.tight_layout(pad=3.0)  # Ajuste do padding entre os gráficos
    plt.show()

    return data, numerical_features


# =====================================
# Função para categorizar as colunas com base na quantidade de valores nulos
# =====================================
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


# =====================================
# Função para matriz de confusao
# =====================================
def plot_confusion_matrix(model, y_pred, y_test, class_names):
    print("classes preditas:", class_names)
    # Se o modelo estiver retornando probabilidades, converta para rótulos
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Computar a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 8))  # Aumentar o tamanho da figura
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()

    # Adicionar rótulos nos eixos
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Mostrar valores na matriz de confusão
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    plt.show()
    
# =====================================
# Função para preprocessar o modelo
# =====================================
def preprocess_data(data):
 # Categorizando as colunas
    print(' - Categorizando as colunas')
    mostly_null, partially_null, no_null = categorize_columns_by_nulls(data)
    print("Colunas com muitos valores nulos:", mostly_null)
    print("Colunas com alguns valores nulos:", partially_null)
    print("Colunas sem valores nulos:", no_null)

    """
    # Tratamento de colunas altamente correlacionadas
    data = remover_colunas_correlacionadas(data, threshold=0.95)
    """    
        
    # Remove a coluna categórica 'sex'
    # porque o sexo não é relevante 
    # para classificar a especie
    data = data.drop(columns=["sex"], errors="ignore")
    
    """
    # Remoção de outliers
    antes = data.shape[0]
    print("Números de linhas antes da remoção dos outliers:", antes)
    data = remover_outliers_iqr(data)
    depois = data.shape[0]
    print("Números de linhas depois da remoção dos outliers:", depois)
    if depois == antes:
        print("Não encontrou outliers")
    else:
        print("Outliers encontrados e removidos:", antes - depois)
    """
    
    return data

# =====================================
# Função principal
# =====================================
def main():
    # Carregamento dos dados
    data = pd.read_csv("/Users/davideneas/Library/CloudStorage/OneDrive-InstitutodeComputação-UniversidadeFederaldeAlagoas/4. Quarto Periodo/PM/atividades/mlclass/03_Validation/abalone_dataset.csv")
    print("Dados de treinamento:")
    print(data.head())
    
    data=preprocess_data(data)
    # Preparando os dados para o modelo
    X = data.drop(columns=["type"])  # Remove a coluna 'type' porque é a variável alvo
    y = data["type"] 
    # Ajustando os valores de y para começar de 0
    y = y - 1
    print("Dados de entrada (y) ajustados:", y)
    """
    Não precisa normalizar assim porque estamos usando arvore de decisão, analisando cada parametro por vez
    # Normalização dos dados para que todas as características fiquem na mesma escala
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X)
    """
    
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modelos para testar: RandomForest
    #n_estimators é o número de árvores na floresta
    # random_state é usado para garantir a reprodutibilidade
    data.info()
    """
    n_estimators= C(n,2)=n!/(2!(n-2)!)
    n_estimators=21
    logo, 21 é a quantidade MAXIMA de arvores de decisão que podemos ter
    levando em consideração que temos 7 tipos de atributos, combinação.
    """
    model = RandomForestClassifier(
        n_estimators=21,       # Aumentando o número de árvores
        max_depth=20,           # Definindo a profundidade máxima
        min_samples_split=10,   # Definindo um valor maior para a divisão
        min_samples_leaf=5,     # Número mínimo de amostras por folha
        max_features='sqrt',    # Usando a raiz quadrada do número de features
        random_state=42
    )
    """
        {
        #  "SVM": SVC(kernel='rbf', random_state=42),
        #  "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        #  "XGBoost": xgb.XGBClassifier(random_state=42)
        }
    """

    # Testar e avaliar todos os modelos
    print(f"\nTreinando Ramdom Forest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo Ramdom Forest: {accuracy:.2f}")
    plot_confusion_matrix(model, y_test, y_pred,  y_test.unique())

    # Validação cruzada
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores para Ramdom Forest: {cv_scores}")
    print(f"Média da validação cruzada: {cv_scores.mean():.2f}")

    # Aplicando o modelo e enviando para o servidor
    print(' - Aplicando modelo e enviando para o servidor')
    data_app = pd.read_csv("/Users/davideneas/Library/CloudStorage/OneDrive-InstitutodeComputação-UniversidadeFederaldeAlagoas/4. Quarto Periodo/PM/atividades/mlclass/03_Validation/abalone_app.csv")

    # Garantir que as colunas de X estejam no data_app
    data_app = preprocess_data(data_app)
    
    # Enviando previsões realizadas com o modelo para o servidor
    URL = "https://aydanomachado.com/mlclass/03_Validation.php"

    # Substitua pela sua chave aqui
    DEV_KEY = "Eneas"

    # Aplicando o modelo e gerando as previsões para o data_app
    y_pred_app = model.predict(data_app)  # Gerando previsões a partir do data_app

    # Convertendo as previsões para o formato JSON
    data = {'dev_key': DEV_KEY,
            'predictions': pd.Series(y_pred_app).to_json(orient='values')}  # Convertendo as previsões para JSON

    # Enviando a requisição e salvando o objeto resposta
    r = requests.post(url=URL, data=data)
    
    print(" - Resposta do servidor:\n", r.text, "\n")  
      
# Executa o script
if __name__ == "__main__":
    main()