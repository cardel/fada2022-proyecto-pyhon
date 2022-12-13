import numpy as np
from load_file import LoadFile

class sparse_matrix_coordinate_format:

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
        #No utilizar self.matriz[i,j] debe sacarse de la representación
        raise NotImplementedError

    def get_row(self, i):
        #No usar self.matrix aqui debe sacarse de la representación
        raise NotImplementedError

    def get_column(self, j):
        # No usar self.matrix aqui debe sacarse de la representación
        raise NotImplementedError

    def set_value(self, i, j, value):
        raise NotImplementedError


    def get_squared_matrix(self):
        instancia = sparse_matrix_coordinate_format()
        """
        Codificar aqui, borrar comentario
        """
        return instancia

    def get_transpose_matrix(self):
        instancia = sparse_matrix_coordinate_format()
        """
        Codificar aqui, borrar comentario
        """
        return instancia