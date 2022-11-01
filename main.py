import numpy as np
import pandas as pd


def get_matrix_for_user(dataset_row):
    result_matrix = {}
    row_states = np.array(dataset_row.split(';'))
    for i in range(0, row_states.size - 1, 1):
        try:
            result_matrix[row_states[i]][row_states[i + 1]] += 1
        except KeyError:
            try:
                result_matrix[row_states[i]][row_states[i + 1]] = 1
            except KeyError:
                result_matrix[row_states[i]] = dict()
                result_matrix[row_states[i]][row_states[i + 1]] = 1
    fill_matrix_prob(result_matrix)
    result_matrix['start'] = row_states[0]
    return result_matrix


def fill_matrix_prob(matrix):
    for row_dict in matrix.items():
        row_sum = 0
        for value in row_dict[1].values():
            row_sum += value
        for value in row_dict[1].items():
            matrix[row_dict[0]][value[0]] = float(value[1]) / row_sum


def get_decision_array_for_user(user, dataset_for_user, related_prob_matrix, gamma=0.05):
    decision_map = {}
    decision_matrix = []
    row_states = np.array(dataset_for_user.split(';'))

    if row_states[0] == related_prob_matrix['start']:
        decision_matrix.append(False)

    for i in range(0, row_states.size - 1, 1):
        try:
            related_prob_matrix[row_states[i]][row_states[i + 1]]
        except KeyError:
            decision_matrix.append(True)
            continue
        decision = np.abs(related_prob_matrix[row_states[i]][row_states[i + 1]] < gamma)
        decision_matrix.append(decision)
    print('\n')
    decision_map[user] = decision_matrix
    return decision_map


data                    = pd.read_csv("datasets/data.txt", sep=':')
dataset_column          = data['DATASET']

matrices = np.vectorize(get_matrix_for_user)(dataset_column)

dataset_for_users = dict()
for i in range(matrices.size):
    dataset_for_users[data['USER'][i]] = matrices[i]

data_true = pd.read_csv("datasets/data_true.txt", sep=':')
data_true_result = []
for i in range(data_true.shape[0]):
    data_true_result.append(get_decision_array_for_user(data_true['USER'][i], data_true['DATASET'][i], dataset_for_users[data_true['USER'][i]], 0.05))

data_fake_result = []
data_fake = pd.read_csv("datasets/data_fake.txt", sep=':')
for i in range(data_fake.shape[0]):
    data_fake_result.append(get_decision_array_for_user(data_true['USER'][i], data_fake['DATASET'][i], dataset_for_users[data_fake['USER'][i]], 0.05))

print(f'Data_True: {data_true_result}')
print(f'Data_Fake: {data_fake_result}')
