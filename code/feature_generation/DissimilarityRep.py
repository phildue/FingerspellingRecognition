from numpy import linalg, zeros


def get_dissim_rep(data):
    dissim_matrix = zeros(shape=(data.shape[1], data.shape[1]))
    for i, object_i in enumerate(data):
        for j, object_j in enumerate(data):
            dissim_matrix[i, j] = get_dist(object_i, object_j)


def get_dist(object_1, object_2):
    #euclidian distance
    return linalg.norm(object_1, object_2)
