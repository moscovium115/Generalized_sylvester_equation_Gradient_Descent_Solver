import numpy as np

def Sylvester_Product(Mat_list_1,Mat_list_2,X_k,transpose=False):
    p=len(Mat_list_1)
    result_matrix=0
    if not transpose:
        for i in range(p):
            # print("debug shapes sylv:",Mat_list_1[i].shape,X_k.shape,Mat_list_2[i].shape)
            result_matrix+=Mat_list_1[i]@X_k@Mat_list_2[i]
    else:
        for i in range(p):
            result_matrix+=Mat_list_1[i].T@X_k@Mat_list_2[i].T
    return result_matrix
  
def Gen_Sylv_gradient_descent(Mat_1,Mat_2,E,iterations):
    """Calculates the solution of the generalized Sylvester equation sum_t=1^p A_t X B_t = E using
    the gradient descent method as described in the paper by Adisorn Kittisopaporn and Pattrawut Chansangiam
    (https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-021-03427-4#Sec3).
    This will be used to construct a solution to the sylvester equation when p>2. As there are no direct methods for this case.
    Mat_1 should be a list containing the matrices A_t, Mat_2 should do the same for the matrices B_t and E is the RHS matrix.
    """
    #initial guess of the solution
    X_0=np.zeros(E.shape)
    X_k=X_0
    for k in range(iterations):
        R_k=E-Sylvester_Product(Mat_1,Mat_2,X_k)
        # print(R_k)
        # print("Residual norm:",np.linalg.norm(R_k))
        W_k=Sylvester_Product(Mat_1,Mat_2,R_k,transpose=True)
        # print(W_k)
        tau_k_1=np.linalg.norm(W_k)/np.linalg.norm(Sylvester_Product(Mat_1,Mat_2,W_k))
        tau_k_1=tau_k_1**2
        # print(tau_k_1)
        X_k=X_k+tau_k_1*W_k
    print("Residual norm:",np.linalg.norm(R_k))
    return X_k
