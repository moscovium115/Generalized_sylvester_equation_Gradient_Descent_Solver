# Generalized_sylvester_equation_Gradient_Descent_Solver
This is a python implementation of the algorithm as described by Adisorn Kittisopaporn and Pattrawut Chansangiam     (https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-021-03427-4#Sec3).
They derive an algorithm that uses gradient descent to solve the following generalized Sylvester Equation

**The Generalized Sylvester Equation**
$$\sum_{t=1}^{p} A_t X B_t =E $$

where A_t, B_t and E are all dense matrices. For p>2, there is no known direct method to obtain the solution matrix X. 
Therefore iterative methodes are needed. Feel free to use this. I hope this code serves to be useful to anyone.
