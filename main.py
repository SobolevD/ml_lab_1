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
    return result_matrix


def fill_matrix_prob(matrix):
    for row_dict in matrix.items():
        row_sum = 0
        for value in row_dict[1].values():
            row_sum += value
        for value in row_dict[1].items():
            matrix[row_dict[0]][value[0]] = float(value[1]) / row_sum


def print_decision_values_for_user(dataset_for_user, related_prob_matrix, gamma=0.05):
    row_states = np.array(dataset_for_user.split(';'))
    for i in range(0, row_states.size - 1, 1):
        try:
            related_prob_matrix[row_states[i]][row_states[i + 1]]
        except KeyError:
            print('False', end=' ')
            continue
        print(f'{np.abs(related_prob_matrix[row_states[i]][row_states[i + 1]] > gamma)}', end=' ')
    print('\n')


data                    = pd.read_csv("datasets/data.txt", sep=':')
dataset_column          = data['DATASET']

matrices = np.vectorize(get_matrix_for_user)(dataset_column)

dataset_for_users = dict()
for i in range(matrices.size):
    dataset_for_users[data['USER'][i]] = matrices[i]

data_true = pd.read_csv("datasets/data_true.txt", sep=':')
for i in range(data_true.shape[0]):
    print(f"Decisions for {data_true['USER'][i]}: ", end=' ')
    print_decision_values_for_user(data_true['DATASET'][i], dataset_for_users[data_true['USER'][i]], 0.01)

data_fake = pd.read_csv("datasets/data_fake.txt", sep=':')
for i in range(data_fake.shape[0]):
    print(f"Decisions for {data_fake['USER'][i]}: ", end=' ')
    print_decision_values_for_user(data_fake['DATASET'][i], dataset_for_users[data_fake['USER'][i]], 0.01)
