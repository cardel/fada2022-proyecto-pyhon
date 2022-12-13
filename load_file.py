#!/usr/bin/env python
#
# Class for loading matrix representation files.
#
import numpy as np

class LoadFile:
    instance_ = None
    matrix = None

    @classmethod
    def getInstance(cls):
        return LoadFile() if (cls.instance_ is None) else cls.instance_

    def loadFile(self, path):
        with open(file=path, encoding="utf-8", mode="r") as input_file:
            first_line = input_file.readline().split(" ")
            rows = int(first_line[0])
            i = 0
            output_matrix = []
            while i < rows:
                output_matrix.append([int(x) for x in input_file.readline().split(" ")])
                i += 1

            self.matrix = np.array(output_matrix)

