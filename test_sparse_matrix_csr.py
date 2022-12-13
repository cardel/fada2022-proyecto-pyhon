from sparse_matrix_csr import sparse_matrix_csr
import numpy as np

file0 = "inputs/matrix0.in"
file1 = "inputs/matrix1.in"
file2 = "inputs/matrix2.in"
file3 = "inputs/matrix3.in"
file4 = "inputs/matrix4.in"
file5 = "inputs/matrix5.in"


def test_create_representation():
    instance_ = sparse_matrix_csr()
    #Test 0
    instance_.create_representation(file0)
    rows = np.array([0, 2, 5, 6, 6, 8, 10, 11, 13])
    cols = np.array([1, 6, 1, 2, 5, 3, 0, 5, 0, 1, 0, 2, 5])
    values = np.array([2, 4, 8, 9, 1, 3, 5, 6, 1, 2, 4, 7, 11])
    assert (rows == instance_.rows).all()
    assert (cols == instance_.columns).all()
    assert (values == instance_.values).all()

    # Test 1
    instance_.create_representation(file1)
    rowsA = np.array([0, 2, 3, 6, 7, 10])
    colsA = np.array([1, 2, 1, 1, 2, 5, 2, 1, 2, 7])
    valuesA = np.array([1, 4, 1, 1, 2, 8, 3, 1, 4, 9])
    assert (rowsA == instance_.rows).all()
    assert (colsA == instance_.columns).all()
    assert (valuesA == instance_.values).all()

    # Test 2
    instance_.create_representation(file2)
    rowsB = np.array([0, 2, 3, 6, 7, 10, 12, 15, 16, 19, 20])
    colsB = np.array([1, 2, 1, 1, 2, 5, 2, 1, 2, 7, 4, 5, 0, 1, 2, 6, 0, 1, 6, 0])
    valuesB = np.array([1, 4, 1, 1, 2, 8, 3, 1, 4, 9, 7, 3, 1, 2, 3, 9, 1, 1, 7, 9])
    assert (rowsB == instance_.rows).all()
    assert (colsB == instance_.columns).all()
    assert (valuesB == instance_.values).all()

    # Test 3
    instance_.create_representation(file3)
    rowsC = np.array([0, 2, 5, 8, 10, 14, 17, 21, 23, 27, 29])
    colsC = np.array([1, 2, 1, 8, 9, 1, 2, 5, 2, 10, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 6, 9, 0, 1, 6, 11, 0, 10])
    valuesC = np.array([1, 4, 1, 1, 1, 1, 2, 8, 3, 3, 1, 4, 9, 9, 7, 3, 7, 1, 2, 3, 6, 9, 9, 1, 1, 7, 1, 9, 4])
    assert (rowsC == instance_.rows).all()
    assert (colsC == instance_.columns).all()
    assert (valuesC == instance_.values).all()

    # Test 4
    instance_.create_representation(file4)
    rowsD = np.array([0, 2, 5, 8, 10, 14, 17, 21, 23, 27, 29, 29, 32, 34, 41, 47])
    colsD = np.array([1, 2, 1, 8, 9, 1, 2, 5, 2, 10, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 6, 9, 0, 1, 6, 11, 0, 10, 3, 5, 6, 1, 2,
             0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 7, 8, 9])
    valuesD = np.array([1, 4, 1, 1, 1, 1, 2, 8, 3, 3, 1, 4, 9, 9, 7, 3, 7, 1, 2, 3, 6, 9, 9, 1, 1, 7, 1, 9, 4, 4, 3, 6, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    assert (rowsD == instance_.rows).all()
    assert (colsD == instance_.columns).all()
    assert (valuesD == instance_.values).all()

    # Test 5
    instance_.create_representation(file5)
    rowsE = np.array([0, 2, 6, 10, 14, 18, 21, 27, 30, 35, 38, 38, 42, 45, 53, 59])
    colsE = np.array([1, 2, 1, 8, 9, 14, 1, 2, 5, 14, 2, 10, 13, 14, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 13, 14, 6, 9, 13, 0, 1,
             6, 11, 14, 0, 10, 13, 3, 5, 6, 13, 1, 2, 13, 0, 1, 2, 3, 4, 5, 6, 13, 0, 1, 2, 7, 8, 9])
    valuesE = np.array([1, 4, 1, 1, 1, 1, 1, 2, 8, 2, 3, 3, 2, 4, 1, 4, 9, 9, 7, 3, 7, 1, 2, 3, 6, 7, 3, 9, 9, 7, 1, 1, 7, 1, 1,
               9, 4, 2, 4, 3, 6, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2])
    assert (rowsE == instance_.rows).all()
    assert (colsE == instance_.columns).all()
    assert (valuesE == instance_.values).all()


def test_getElement():
    instance_ = sparse_matrix_csr()
    # Test 1
    instance_.create_representation(file1)
    assert instance_.get_element(0, 0) == 0
    assert instance_.get_element(1, 2) == 0
    assert instance_.get_element(4, 7) == 9
    # Test 2
    instance_.create_representation(file2)
    assert instance_.get_element(0, 0) == 0
    assert instance_.get_element(5, 3) == 0
    assert instance_.get_element(6, 2) == 3
    # Test 3
    instance_.create_representation(file3)
    assert instance_.get_element(0, 0) == 0
    assert instance_.get_element(4, 10) == 9
    assert instance_.get_element(9, 9) == 0
    # Test 4
    instance_.create_representation(file4)
    assert instance_.get_element(0, 0) == 0
    assert instance_.get_element(4, 10) == 9
    assert instance_.get_element(14, 8) == 2
    # Test 5
    instance_.create_representation(file5)
    assert instance_.get_element(0, 0) == 0
    assert instance_.get_element(4, 10) == 9
    assert instance_.get_element(14, 14) == 0


def test_getRow():
    instance_ = sparse_matrix_csr()
    # Test 1
    instance_.create_representation(file1)
    assert (instance_.get_row(2) == np.array(np.array([0, 1, 2, 0, 0, 8, 0, 0]))).all()
    assert (instance_.get_row(4) == np.array(np.array([0, 1, 4, 0, 0, 0, 0, 9]))).all()

    # Test 2
    instance_.create_representation(file2)
    assert (instance_.get_row(5) == np.array(np.array([0, 0, 0, 0, 7, 3, 0, 0]))).all()
    assert (instance_.get_row(8) == np.array(np.array([1, 1, 0, 0, 0, 0, 7, 0]))).all()

    # Test 3
    instance_.create_representation(file3)
    assert (instance_.get_row(5) == np.array(np.array([0, 0, 0, 0, 7, 3, 0, 0, 0, 0, 7, 0]))).all()
    assert (instance_.get_row(9) == np.array(np.array([9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0]))).all()

    # Test 4
    instance_.create_representation(file4)
    assert (instance_.get_row(9) == np.array(np.array([9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0]))).all()
    assert (instance_.get_row(14) == np.array(np.array([2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0]))).all()

    # Test 5
    instance_.create_representation(file5)
    assert (instance_.get_row(2) == np.array(np.array([0, 1, 2, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2]))).all()
    assert (instance_.get_row(13) == np.array(np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 0]))).all()


def test_getColumn():
    instance_ = sparse_matrix_csr()
    # Test 1
    instance_.create_representation(file1)
    assert (instance_.get_column(1) == np.array(np.array([1, 1, 1, 0, 1]))).all()
    assert (instance_.get_column(4) == np.array(np.array([0, 0, 0, 0, 0]))).all()
    # Test 2
    instance_.create_representation(file2)
    assert (instance_.get_column(5) == np.array(np.array([0, 0, 8, 0, 0, 3, 0, 0, 0, 0]))).all()
    assert (instance_.get_column(6) == np.array(np.array([0, 0, 0, 0, 0, 0, 0, 9, 7, 0]))).all()

    # Test 3
    instance_.create_representation(file3)
    assert (instance_.get_column(5) == np.array(np.array([0, 0, 8, 0, 0, 3, 0, 0, 0, 0]))).all()
    assert (instance_.get_column(9) == np.array(np.array([0, 1, 0, 0, 0, 0, 0, 9, 0, 0]))).all()

    # Test 4
    instance_.create_representation(file4)
    assert (instance_.get_column(9) == np.array(np.array([0, 1, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 2]))).all()
    assert (instance_.get_column(10) == np.array(np.array([0, 0, 0, 3, 9, 7, 6, 0, 0, 4, 0, 0, 0, 0, 0]))).all()

    # Test 5
    instance_.create_representation(file5)
    assert (instance_.get_column(2) == np.array(np.array([4, 0, 2, 3, 4, 0, 3, 0, 0, 0, 0, 0, 1, 1, 2]))).all()
    assert (instance_.get_column(13) == np.array(np.array([0, 0, 0, 2, 0, 0, 7, 7, 0, 2, 0, 9, 2, 3, 0]))).all()


def test_setValue():
    instance_ = sparse_matrix_csr()
    # Test 0
    instance_.create_representation(file0)
    instance_.set_value(0, 4, 10)
    rows = np.array([0, 3, 6, 7, 7, 9, 11, 12, 14])
    cols = np.array([1, 4, 6, 1, 2, 5, 3, 0, 5, 0, 1, 0, 2, 5])
    values = np.array([2, 10, 4, 8, 9, 1, 3, 5, 6, 1, 2, 4, 7, 11])
    assert (rows == instance_.rows).all()
    assert (cols == instance_.columns).all()
    assert (values == instance_.values).all()

    # Test 1
    instance_.create_representation(file1)
    instance_.set_value(0, 4, 10)
    rowsA = np.array([0, 3, 4, 7, 8, 11])
    colsA = np.array([1, 2, 4, 1, 1, 2, 5, 2, 1, 2, 7])
    valuesA = np.array([1, 4, 10, 1, 1, 2, 8, 3, 1, 4, 9])
    assert (rowsA == instance_.rows).all()
    assert (colsA == instance_.columns).all()
    assert (valuesA == instance_.values).all()

    # Test 2
    instance_.create_representation(file2)
    instance_.set_value(0, 4, 10)
    rowsB = np.array([0, 3, 4, 7, 8, 11, 13, 16, 17, 20, 21])
    colsB = np.array([1, 2, 4, 1, 1, 2, 5, 2, 1, 2, 7, 4, 5, 0, 1, 2, 6, 0, 1, 6, 0])
    valuesB = np.array([1, 4, 10, 1, 1, 2, 8, 3, 1, 4, 9, 7, 3, 1, 2, 3, 9, 1, 1, 7, 9])

    assert (rowsB == instance_.rows).all()
    assert (colsB == instance_.columns).all()
    assert (valuesB == instance_.values).all()

    # Test 3
    instance_.create_representation(file3)
    instance_.set_value(0, 4, 10)
    rowsC = np.array([0, 3, 6, 9, 11, 15, 18, 22, 24, 28, 30])
    colsC = np.array([1, 2, 4, 1, 8, 9, 1, 2, 5, 2, 10, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 6, 9, 0, 1, 6, 11, 0, 10])
    valuesC = np.array([1, 4, 10, 1, 1, 1, 1, 2, 8, 3, 3, 1, 4, 9, 9, 7, 3, 7, 1, 2, 3, 6, 9, 9, 1, 1, 7, 1, 9, 4])

    assert (rowsC == instance_.rows).all()
    assert (colsC == instance_.columns).all()
    assert (valuesC == instance_.values).all()

    # Test 4
    instance_.create_representation(file4)
    instance_.set_value(0, 4, 10)
    rowsD = np.array([0, 3, 6, 9, 11, 15, 18, 22, 24, 28, 30, 30, 33, 35, 42, 48])
    colsD = np.array([1, 2, 4, 1, 8, 9, 1, 2, 5, 2, 10, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 6, 9, 0, 1, 6, 11, 0, 10, 3, 5, 6, 1,
             2, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 7, 8, 9])
    valuesD = np.array([1, 4, 10, 1, 1, 1, 1, 2, 8, 3, 3, 1, 4, 9, 9, 7, 3, 7, 1, 2, 3, 6, 9, 9, 1, 1, 7, 1, 9, 4, 4, 3, 6, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    assert (rowsD == instance_.rows).all()
    assert (colsD == instance_.columns).all()
    assert (valuesD == instance_.values).all()

    # Test 5
    instance_.create_representation(file5)
    instance_.set_value(0, 4, 10)
    rowsE = np.array([0, 3, 7, 11, 15, 19, 22, 28, 31, 36, 39, 39, 43, 46, 54, 60])
    colsE = np.array([1, 2, 4, 1, 8, 9, 14, 1, 2, 5, 14, 2, 10, 13, 14, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 13, 14, 6, 9, 13, 0,
             1, 6, 11, 14, 0, 10, 13, 3, 5, 6, 13, 1, 2, 13, 0, 1, 2, 3, 4, 5, 6, 13, 0, 1, 2, 7, 8, 9])
    valuesE = np.array([1, 4, 10, 1, 1, 1, 1, 1, 2, 8, 2, 3, 3, 2, 4, 1, 4, 9, 9, 7, 3, 7, 1, 2, 3, 6, 7, 3, 9, 9, 7, 1, 1, 7, 1,
               1, 9, 4, 2, 4, 3, 6, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2])
    assert (rowsE == instance_.rows).all()
    assert (colsE == instance_.columns).all()
    assert (valuesE == instance_.values).all()


def test_getSquareMatrix():
    instance_ = sparse_matrix_csr()
    #Test 0
    instance_.create_representation(file0)
    inst = instance_.get_squared_matrix()
    rows = np.array([0, 2, 5, 6, 6, 8, 10, 11, 13])
    cols = np.array([1, 6, 1, 2, 5, 3, 0, 5, 0, 1, 0, 2, 5])
    values = np.array([4, 16, 64, 81, 1, 9, 25, 36, 1, 4, 16, 49, 121])
    assert (rows == inst.rows).all()
    assert (cols == inst.columns).all()
    assert (values == inst.values).all()

    # Test 1
    instance_.create_representation(file1)
    inst = instance_.get_squared_matrix()

    rowsA = np.array([0, 2, 3, 6, 7, 10])
    colsA = np.array([1, 2, 1, 1, 2, 5, 2, 1, 2, 7])
    valuesA = np.array([1, 16, 1, 1, 4, 64, 9, 1, 16, 81])
    assert (rowsA == inst.rows).all()
    assert (colsA == inst.columns).all()
    assert (valuesA == inst.values).all()

    # Test 2
    instance_.create_representation(file2)
    inst = instance_.get_squared_matrix()

    rowsB = np.array([0, 2, 3, 6, 7, 10, 12, 15, 16, 19, 20])
    colsB = np.array([1, 2, 1, 1, 2, 5, 2, 1, 2, 7, 4, 5, 0, 1, 2, 6, 0, 1, 6, 0])
    valuesB = np.array([1, 16, 1, 1, 4, 64, 9, 1, 16, 81, 49, 9, 1, 4, 9, 81, 1, 1, 49, 81])
    assert (rowsB == inst.rows).all()
    assert (colsB == inst.columns).all()
    assert (valuesB == inst.values).all()

    # Test 3
    instance_.create_representation(file3)
    inst = instance_.get_squared_matrix()

    rowsC = np.array([0, 2, 5, 8, 10, 14, 17, 21, 23, 27, 29])
    colsC = np.array([1, 2, 1, 8, 9, 1, 2, 5, 2, 10, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 6, 9, 0, 1, 6, 11, 0, 10])
    valuesC = np.array([1, 16, 1, 1, 1, 1, 4, 64, 9, 9, 1, 16, 81, 81, 49, 9, 49, 1, 4, 9, 36, 81, 81, 1, 1, 49, 1, 81, 16])
    assert (rowsC == inst.rows).all()
    assert (colsC == inst.columns).all()
    assert (valuesC == inst.values).all()

    # Test 4
    instance_.create_representation(file4)
    inst = instance_.get_squared_matrix()

    rowsD = np.array([0, 2, 5, 8, 10, 14, 17, 21, 23, 27, 29, 29, 32, 34, 41, 47])
    colsD = np.array([1, 2, 1, 8, 9, 1, 2, 5, 2, 10, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 6, 9, 0, 1, 6, 11, 0, 10, 3, 5, 6, 1, 2,
             0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 7, 8, 9])
    valuesD = np.array([1, 16, 1, 1, 1, 1, 4, 64, 9, 9, 1, 16, 81, 81, 49, 9, 49, 1, 4, 9, 36, 81, 81, 1, 1, 49, 1, 81, 16, 16, 9, 36, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4])

    assert (rowsD == inst.rows).all()
    assert (colsD == inst.columns).all()
    assert (valuesD == inst.values).all()

    # Test 5
    instance_.create_representation(file5)
    inst = instance_.get_squared_matrix()

    rowsE = np.array([0, 2, 6, 10, 14, 18, 21, 27, 30, 35, 38, 38, 42, 45, 53, 59])
    colsE = np.array([1, 2, 1, 8, 9, 14, 1, 2, 5, 14, 2, 10, 13, 14, 1, 2, 7, 10, 4, 5, 10, 0, 1, 2, 10, 13, 14, 6, 9, 13, 0, 1,
             6, 11, 14, 0, 10, 13, 3, 5, 6, 13, 1, 2, 13, 0, 1, 2, 3, 4, 5, 6, 13, 0, 1, 2, 7, 8, 9])
    valuesE = np.array([1, 16, 1, 1, 1, 1, 1, 4, 64, 4, 9, 9, 4, 16, 1, 16, 81, 81, 49, 9, 49, 1, 4, 9, 36, 49, 9, 81, 81, 49, 1, 1, 49, 1, 1, 81, 16, 4, 16, 9, 36, 81, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 9, 4, 4, 4, 4, 4, 4])
    assert (rowsE == inst.rows).all()
    assert (colsE == inst.columns).all()
    assert (valuesE == inst.values).all()


def test_getTransposedMatrix():
    instance_ = sparse_matrix_csr()
    instance_.create_representation(file1)
    inst = instance_.get_transpose_matrix()

    # Test 0
    instance_.create_representation(file0)
    inst = instance_.get_transpose_matrix()
    rows = np.array([0, 3, 6, 8, 9, 9, 12, 13])
    cols = np.array([4, 5, 6, 0, 1, 5, 1, 7, 2, 1, 4, 7, 0])
    values = np.array([5, 1, 4, 2, 8, 2, 9, 7, 3, 1, 6, 11, 4])
    assert (rows == inst.rows).all()
    assert (cols == inst.columns).all()
    assert (values == inst.values).all()

    # Test 1
    instance_.create_representation(file1)
    inst = instance_.get_transpose_matrix()
    rowsA = np.array([0, 0, 4, 8, 8, 8, 9, 9, 10])
    colsA = np.array([0, 1, 2, 4, 0, 2, 3, 4, 2, 4])
    valuesA = np.array([1, 1, 1, 1, 4, 2, 3, 4, 8, 9])
    assert (rowsA == inst.rows).all()
    assert (colsA == inst.columns).all()
    assert (valuesA == inst.values).all()

    # Test 2
    instance_.create_representation(file2)
    inst = instance_.get_transpose_matrix()
    rowsB = np.array([0, 3, 9, 14, 14, 15, 17, 19, 20])
    colsB = np.array([6, 8, 9, 0, 1, 2, 4, 6, 8, 0, 2, 3, 4, 6, 5, 2, 5, 7, 8, 4])
    valuesB = np.array([1, 1, 9, 1, 1, 1, 1, 2, 1, 4, 2, 3, 4, 3, 7, 8, 3, 9, 7, 9])

    assert (rowsB == inst.rows).all()
    assert (colsB == inst.columns).all()
    assert (valuesB == inst.values).all()

    # Test 3
    instance_.create_representation(file3)
    inst = instance_.get_transpose_matrix()
    rowsC = np.array([0, 3, 9, 14, 14, 15, 17, 19, 20, 21, 23, 28, 29])
    colsC = np.array([6, 8, 9, 0, 1, 2, 4, 6, 8, 0, 2, 3, 4, 6, 5, 2, 5, 7, 8, 4, 1, 1, 7, 3, 4, 5, 6, 9, 8])
    valuesC = np.array([1, 1, 9, 1, 1, 1, 1, 2, 1, 4, 2, 3, 4, 3, 7, 8, 3, 9, 7, 9, 1, 1, 9, 3, 9, 7, 6, 4, 1])

    assert (rowsC == inst.rows).all()
    assert (colsC == inst.columns).all()
    assert (valuesC == inst.values).all()

    # Test 4
    instance_.create_representation(file4)
    inst = instance_.get_transpose_matrix()
    rowsD = np.array([0, 5, 14, 22, 24, 26, 30, 34, 36, 38, 41, 46, 47])
    colsD = np.array([6, 8, 9, 13, 14, 0, 1, 2, 4, 6, 8, 12, 13, 14, 0, 2, 3, 4, 6, 12, 13, 14, 11, 13, 5, 13, 2, 5, 11, 13, 7, 8, 11, 13, 4, 14, 1, 14, 1, 7, 14, 3, 4, 5, 6, 9, 8])
    valuesD = np.array([1, 1, 9, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 4, 2, 3, 4, 3, 1, 1, 2, 4, 1, 7, 1, 8, 3, 3, 1, 9, 7, 6, 1, 9, 2, 1, 2, 1, 9, 2, 3, 9, 7, 6, 4, 1])

    assert (rowsD == inst.rows).all()
    assert (colsD == inst.columns).all()
    assert (valuesD == inst.values).all()

    # Test 5
    instance_.create_representation(file5)
    inst = instance_.get_transpose_matrix()
    rowsE = np.array([0, 5, 14, 22, 24, 26, 30, 34, 36, 38, 41, 46, 47, 47, 54, 59])
    colsE = np.array([6, 8, 9, 13, 14, 0, 1, 2, 4, 6, 8, 12, 13, 14, 0, 2, 3, 4, 6, 12, 13, 14, 11, 13, 5, 13, 2, 5, 11, 13, 7, 8, 11, 13, 4, 14, 1, 14, 1, 7, 14, 3, 4, 5, 6, 9, 8, 3, 6, 7, 9, 11, 12, 13, 1, 2, 3, 6, 8])
    valuesE = np.array([1, 1, 9, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 4, 2, 3, 4, 3, 1, 1, 2, 4, 1, 7, 1, 8, 3, 3, 1, 9, 7, 6, 1, 9, 2, 1, 2, 1, 9, 2, 3, 9, 7, 6, 4, 1, 2, 7, 7, 2, 9, 2, 3, 1, 2, 4, 3, 1])
    assert (rowsE == inst.rows).all()
    assert (colsE == inst.columns).all()
    assert (valuesE == inst.values).all()
