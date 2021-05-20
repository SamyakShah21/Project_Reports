# ME 766 High Performance Scientific Computing- Final Project  
# In the Moment of Heat
<br>

## Completed by - 
* Akshit Srivastava (180110008)
* Rohan Pipersenia (180110064)
* Samyak Shah (18D070062)
* Souvik Pal (18D110011)

## Introduction
We solve the following problem:

> _A square plate [−1, 1] × [−1, 1] is at temperature u = 0. At time *t* = 0
the temperature is increased to *u = 5* along one of the four sides while
being held at *u = 0* along the other three sides, and heat then flows into
the plate according to *u_t = ∆u*_
>
>_When does the temperature reach *u = 1* at the center of the plate?_
>
> In the Moment of Heat, **SIAM 100-Digit Challenge**

The solution is achieved by way of implementing a Crank-Nicholson, central difference
finite difference scheme on a 2-dimensional square of the above dimensions. The
scheme is implemented _via_ a combination of the techniques listed here:
* Linear Operators (OpenCL kernels) to apply the numerical scheme
* The Conjugate Gradient method to apply the inverse of a matrix
* A Preconditioner matrix formed through Sparse Incomplete LU decomposition
* Numba-accelerated Python for functions that do simple calculations
* The Secant Method for finding the solution to the problem

The **code** directory contains the project code.  
**report.pdf** contains a comprehensive report for the project.  
**slides.pptx** contains the presentation slides.