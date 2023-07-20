import numpy as np
from GD_solver import *

Mat_1=[np.array([[1,2],[3,4]]),np.eye(2),np.array([[5,6],[7,8]])]
Mat_2=[np.eye(2),np.array([[5,6],[7,10]]),np.array([[1,2],[3,4]])]
iterations=100
E=np.array([[3,8],[9,10]])
solution_matrix=Gen_Sylv_gradient_descent(Mat_1,Mat_2,E,iterations)
print(Sylvester_Product(Mat_1,Mat_2,solution_matrix))
print(E)
