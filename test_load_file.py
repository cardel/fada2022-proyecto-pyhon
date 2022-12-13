from load_file import LoadFile
import pytest
import numpy as np

loader = LoadFile.getInstance()
file0 = "inputs/matrix0.in"
file1 = "inputs/matrix1.in"
file2 = "inputs/matrix2.in"
file3 = "inputs/matrix3.in"
file4 = "inputs/matrix4.in"
file5 = "inputs/matrix5.in"

def readMatrix(path):
    with open(file=path, encoding="utf-8", mode="r") as input_file:
        first_line = input_file.readline().split(" ")
        rows = int(first_line[0])
        i = 0
        output_matrix = []
        while i < rows:
            output_matrix.append([int(x) for x in input_file.readline().split(" ")])
            i += 1

    return np.array(output_matrix)

def test_loadFile():
    files = [file0, file1, file2, file3, file4, file5]

    for f in files:
        loader.loadFile(f)
        matriz = readMatrix(f)
        assert (loader.matrix == matriz).all()
