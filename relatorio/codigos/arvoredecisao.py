import numpy as np

# Função para calcular o índice Gini de um grupo de amostras
def gini(groups, classes):
    # Número total de amostras
    n_instances = float(sum([len(group) for group in groups]))
    
    gini_index = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        # Proporção de cada classe no grupo
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini_index += (1.0 - score) * (size / n_instances)
    
    return gini_index

# Função para dividir os dados em dois grupos com base em um valor específico de uma coluna
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Selecionar a melhor divisão do dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini_index = gini(groups, class_values)
            if gini_index < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini_index, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# Criar um nó folha
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Dividir os nós, criando sub-árvores de forma recursiva ou folhas
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # Verificar se não há divisão
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Verificar profundidade máxima
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Processar o nó esquerdo
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # Processar o nó direito
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Construir uma árvore de decisão
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Fazer uma previsão com a árvore de decisão
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Exemplo de uso
if __name__ == '__main__':
    # Dataset de exemplo: [característica1, característica2, ..., classe]
    dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]

    # Definindo os parâmetros da árvore
    max_depth = 3
    min_size = 1

    # Construir a árvore
    tree = build_tree(dataset, max_depth, min_size)

    # Fazer previsões
    for row in dataset:
        prediction = predict(tree, row)
        print('Esperado=%d, Previsto=%d' % (row[-1], prediction))