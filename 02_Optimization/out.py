import requests   # Biblioteca para enviar requisições HTTP (permite conversar com o servidor)
import random     # Biblioteca para gerar números aleatórios
import math       # Biblioteca para operações matemáticas (usada, por exemplo, para a função exponencial)
import time       # Biblioteca para funções relacionadas ao tempo (não utilizada de forma crítica aqui)

# ============================================================
# Função de avaliação: envia requisição ao servidor e retorna o ganho e a resposta completa
# ============================================================
# Essa função "evaluate" é responsável por testar uma configuração da antena.
# A antena é definida por 6 ângulos (phi1, theta1, phi2, theta2, phi3, theta3) que determinam a forma dela.
# A função constrói uma URL com esses valores e envia uma requisição HTTP para o servidor que simula a antena.
# O servidor responde com um texto onde a primeira linha é o "ganho" (um número que indica o desempenho da antena)
# e as demais linhas mostram os ângulos que foram usados.
# Se ocorrer algum erro (por exemplo, se o servidor não responder), a função retorna um ganho muito baixo (-infinito).
def evaluate(angles):
    """
    Avalia uma configuração da antena.
    
    Parâmetros:
      angles: lista com 6 inteiros representando os ângulos [phi1, theta1, phi2, theta2, phi3, theta3]
    
    Retorna:
      gain: ganho (float) retornado pelo servidor; se ocorrer erro, retorna -infinito
      response_text: string com a resposta completa do servidor
    """
    # Monta a URL com os parâmetros (os ângulos) para fazer a requisição ao servidor
    url = (
        f"http://localhost:8080/antenna/simulate?"
        f"phi1={angles[0]}&theta1={angles[1]}&"
        f"phi2={angles[2]}&theta2={angles[3]}&"
        f"phi3={angles[4]}&theta3={angles[5]}"
    )
    try:
        # Envia uma requisição GET ao servidor com timeout de 5 segundos
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # Se a resposta for bem-sucedida (código 200), pega o texto completo da resposta
            response_text = response.text.strip()
            # A primeira linha da resposta contém o ganho obtido com essa configuração
            gain = float(response_text.splitlines()[0])
            return gain, response_text
        else:
            # Se o servidor retornar um erro, imprime uma mensagem e retorna -infinito para o ganho
            print("Erro na requisição:", response.status_code)
            return -math.inf, ""
    except Exception as e:
        # Se ocorrer qualquer exceção (erro), exibe a mensagem e retorna -infinito para o ganho
        print("Erro na requisição:", e)
        return -math.inf, ""

# ============================================================
# Funções Auxiliares
# ============================================================
# Função para gerar uma solução aleatória: uma configuração com 6 ângulos entre 0 e 359 graus.
def random_solution():
    """Gera uma solução aleatória: lista de 6 ângulos inteiros no intervalo [0, 359]."""
    return [random.randint(0, 359) for _ in range(6)]

# Função que gera uma "solução vizinha" (ou seja, uma configuração similar com uma pequena alteração)
# Essa função modifica aleatoriamente um dos 6 ângulos por um valor pequeno (dentro de ±step graus).
def neighbor(solution, step=5):
    """
    Gera uma solução vizinha alterando um ângulo aleatoriamente em ±step graus.
    
    Parâmetros:
      solution: solução atual (lista de 6 inteiros)
      step: variação máxima permitida
      
    Retorna:
      nova solução (lista de 6 inteiros)
    """
    new_solution = solution.copy()  # Cria uma cópia da solução atual para não alterá-la diretamente
    index = random.randint(0, 5)      # Escolhe um dos 6 ângulos aleatoriamente
    delta = random.randint(-step, step)  # Gera um pequeno valor de alteração (positivo ou negativo)
    # Atualiza o ângulo escolhido, usando a operação módulo para garantir que o valor esteja entre 0 e 359
    new_solution[index] = (new_solution[index] + delta) % 360
    return new_solution

# ============================================================
# Algoritmo Hill Climbing (Subida da Encosta)
# ============================================================
# Esse algoritmo é um "agente" que busca melhorar a solução de forma iterativa.
# Ele começa com uma configuração aleatória e, em cada iteração, tenta encontrar uma solução "vizinha"
# (ou seja, uma pequena variação da solução atual) que apresente um ganho melhor.
# Se encontrar, ele passa a usar essa nova solução e continua o processo.
def hill_climbing(max_iter=1000, step=5):
    current = random_solution()  # Solução inicial aleatória
    best_gain, best_response = evaluate(current)  # Avalia essa solução inicial
    best_solution = current.copy()  # Armazena a melhor solução encontrada até o momento
    iterations = 0

    # Executa o processo por um número máximo de iterações
    while iterations < max_iter:
        neighbor_sol = neighbor(current, step)  # Gera uma solução vizinha
        gain, response_text = evaluate(neighbor_sol)  # Avalia a solução vizinha
        # Se a solução vizinha for melhor, atualiza a solução atual e os melhores valores encontrados
        if gain > best_gain:
            current = neighbor_sol
            best_gain = gain
            best_solution = neighbor_sol.copy()
            best_response = response_text
            print(f"Iteração {iterations}: Novo melhor ganho {best_gain} com solução {best_solution}")
        iterations += 1
    return best_solution, best_gain, best_response

# ============================================================
# Algoritmo de Têmpera Simulada (Simulated Annealing)
# ============================================================
# Esse é outro "agente" que busca a melhor solução, mas com uma estratégia que permite, ocasionalmente,
# aceitar soluções piores. Isso ajuda a evitar ficar preso em soluções locais que não são as melhores.
# A técnica utiliza o conceito de "temperatura", que diminui com o tempo. No início, o algoritmo aceita
# mais soluções piores, mas com o tempo fica mais rigoroso.
def simulated_annealing(initial_temp=1000, cooling_rate=0.99, max_iter=1000, step=5):
    current = random_solution()  # Solução inicial aleatória
    current_gain, current_response = evaluate(current)  # Avalia a solução inicial
    best_solution = current.copy()
    best_gain = current_gain
    best_response = current_response
    temp = initial_temp  # Temperatura inicial (controla a probabilidade de aceitar soluções piores)
    iterations = 0

    while iterations < max_iter and temp > 1e-3:  # Continua enquanto houver iterações e a temperatura não esfriar demais
        new_solution = neighbor(current, step)  # Gera uma solução vizinha
        new_gain, new_response = evaluate(new_solution)  # Avalia a nova solução
        delta = new_gain - current_gain  # Diferença de ganho entre a nova solução e a atual

        # Se a nova solução for melhor, ou mesmo se for pior mas for aceita com uma certa probabilidade...
        if delta > 0 or math.exp(delta / temp) > random.random():
            current = new_solution  # Atualiza a solução atual
            current_gain = new_gain
            current_response = new_response
            # Se a nova solução for a melhor até o momento, atualiza os registros
            if new_gain > best_gain:
                best_solution = new_solution.copy()
                best_gain = new_gain
                best_response = new_response
                print(f"Iteração {iterations}: Novo melhor ganho {best_gain} com solução {best_solution}")
        temp *= cooling_rate  # Diminui a temperatura (tornando a aceitação de soluções piores menos provável)
        iterations += 1
    return best_solution, best_gain, best_response

# ============================================================
# Algoritmo Genético
# ============================================================
# Os algoritmos genéticos são inspirados na evolução natural. Aqui, cada "indivíduo" da população
# representa uma configuração da antena (um conjunto de 6 ângulos). Esses indivíduos "evoluem"
# através de processos que simulam seleção natural, cruzamento (combinação de características de dois pais)
# e mutação (pequenas alterações aleatórias). O objetivo é que, ao longo das gerações, a população
# evolua para ter configurações com ganhos cada vez melhores.
def genetic_algorithm(population_size=20, generations=50, mutation_rate=0.1, crossover_rate=0.8):
    # Cria uma população inicial com soluções aleatórias
    population = [random_solution() for _ in range(population_size)]
    fitness = []    # Lista para armazenar os ganhos de cada indivíduo
    responses = []  # Lista para armazenar as respostas completas do servidor para cada indivíduo
    for individual in population:
        g, resp = evaluate(individual)
        fitness.append(g)
        responses.append(resp)
    
    # Encontra o melhor indivíduo da população inicial
    best_index = max(range(population_size), key=lambda i: fitness[i])
    best_solution = population[best_index].copy()
    best_gain = fitness[best_index]
    best_response = responses[best_index]

    # Processo evolutivo: repete por um número de gerações
    for gen in range(generations):
        new_population = []
        # Cria uma nova população combinando e mutando indivíduos
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness)  # Seleciona um pai
            parent2 = tournament_selection(population, fitness)  # Seleciona outro pai
            # Realiza o cruzamento (crossover) com certa probabilidade para combinar os pais
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # Aplica mutação com certa probabilidade para introduzir variações
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            new_population.extend([child1, child2])
        # Atualiza a população com os novos indivíduos (limitando ao tamanho desejado)
        population = new_population[:population_size]
        fitness = []
        responses = []
        # Avalia todos os novos indivíduos
        for individual in population:
            g, resp = evaluate(individual)
            fitness.append(g)
            responses.append(resp)
        # Encontra o melhor indivíduo da nova população
        current_best_index = max(range(population_size), key=lambda i: fitness[i])
        if fitness[current_best_index] > best_gain:
            best_gain = fitness[current_best_index]
            best_solution = population[current_best_index].copy()
            best_response = responses[current_best_index]
            print(f"Geração {gen}: Novo melhor ganho {best_gain} com solução {best_solution}")
    return best_solution, best_gain, best_response

# ============================================================
# Funções de apoio para o Algoritmo Genético
# ============================================================
# Função de seleção por torneio: escolhe o melhor indivíduo dentre um grupo aleatório.
# Essa técnica simula um "torneio" onde os indivíduos competem e o melhor é selecionado.
def tournament_selection(population, fitness, tournament_size=3):
    selected = random.sample(list(zip(population, fitness)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

# Função de cruzamento (crossover): combina duas soluções (pais) para gerar duas novas soluções (filhos).
# É como misturar características dos pais para obter uma nova configuração.
def crossover(parent1, parent2):
    point = random.randint(1, 5)  # Escolhe um ponto de corte (entre 1 e 5) para dividir os pais
    child1 = parent1[:point] + parent2[point:]  # Combina parte do primeiro pai com parte do segundo
    child2 = parent2[:point] + parent1[point:]  # Combina parte do segundo pai com parte do primeiro
    return child1, child2

# Função de mutação: realiza uma pequena alteração aleatória em uma solução.
# Isso ajuda a manter a diversidade e pode permitir encontrar soluções melhores.
def mutate(solution, step=5):
    mutated = solution.copy()
    index = random.randint(0, 5)  # Escolhe aleatoriamente um dos 6 ângulos
    delta = random.randint(-step, step)  # Gera um pequeno ajuste
    mutated[index] = (mutated[index] + delta) % 360  # Atualiza o ângulo garantindo que fique entre 0 e 359
    return mutated

# ============================================================
# Função Principal: executa todos os algoritmos
# ============================================================
# Nesta função, os três algoritmos (Hill Climbing, Têmpera Simulada e Algoritmo Genético) são executados um após o outro.
# Cada um deles tenta encontrar a melhor configuração da antena e, ao final, o resultado (no formato retornado pelo servidor)
# é impresso na tela.
def main():
    print("Executando Hill Climbing...")
    best_solution_hc, best_gain_hc, best_response_hc = hill_climbing()
    print("\nMelhor solução Hill Climbing (formato do servidor):")
    print(best_response_hc)
    
    print("\nExecutando Simulated Annealing...")
    best_solution_sa, best_gain_sa, best_response_sa = simulated_annealing()
    print("\nMelhor solução Simulated Annealing (formato do servidor):")
    print(best_response_sa)
    
    print("\nExecutando Algoritmo Genético...")
    best_solution_ga, best_gain_ga, best_response_ga = genetic_algorithm()
    print("\nMelhor solução Algoritmo Genético (formato do servidor):")
    print(best_response_ga)

# Se o arquivo for executado diretamente, chama a função principal para iniciar os algoritmos
if __name__ == "__main__":
    main()