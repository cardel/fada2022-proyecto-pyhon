import numpy as np
from load_file import LoadFile
class sparse_matrix_csc:

    @classmethod
    def __init__(cls):
        cls.matrix = np.array([])
        cls.rows = None
        cls.columns = None
        cls.values = None
        cls.loader = LoadFile.getInstance()

    def create_representation(self, input_file):
        raise NotImplementedError

    def get_element(self, i, j):
        #No usar self.matrix aqui para retornar el valor de la posición i,j
        raise NotImplementedError

    def get_row(self, i):
        #No usar self.matrix aqui para retornar la fila i
        raise NotImplementedError

    def get_column(self, j):
        # No usar self.matrix aqui para retornar la columna j
        raise NotImplementedError

    def set_value(self, i, j, value):
        raise NotImplementedError

    def get_squared_matrix(self):
        instancia = sparse_matrix_csc()
        """
        Codificar aqui, borrar comentario
        """
        return instancia

    def get_transpose_matrix(self):
        instancia = sparse_matrix_csc()
        """
        Codificar aqui, borrar comentario
        """
        return instancia