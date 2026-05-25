# 2-cofactor.py

minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculates the cofactor matrix"""

    minors = minor(matrix)
    n = len(minors)

    cofactor_matrix = []

    for i in range(n):
        row = []

        for j in range(n):
            row.append(((-1) ** (i + j)) * minors[i][j])

        cofactor_matrix.append(row)

    return cofactor_matrix