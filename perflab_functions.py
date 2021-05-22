# ENGSCI233: Lab - Performance
# perflab_functions.py

# PURPOSE:
# Functions for LU factorisation and matrix multiplication.

# PREPARATION:
# Notebook performance.ipynb.

# SUBMISSION:
# - YOU MUST submit this file to complete the lab. 
# - DO NOT change the file name.

# TO DO:
# - MODIFY the matmult2() and row_reduction() to optimise their run time.
# - COMPLETE read_and_factor()
# - DO NOT modify the other functions.

# imports
import os
import numpy as np
from matplotlib import pyplot as plt

#############################################################################
# LU factorisation functions
#############################################################################

# this function is complete
def lu_factor(A):
    # get dimensions of square matrix 
    n = np.shape(A)[0] 	
    
    # loop over each row in the matrix
    for i in range(n-1):		
        
        # Step 0: Get the pivot value
        pivot_value = A[i,i]											

        # perform row reduction operations
        # in the previous lab, this code was not contained within a function
        A = row_reduction(A, i, n, pivot_value)
        
    return A

# **this function should be modified**
#                           --------
def row_reduction(A, i, n, pivot_value):
    """
    Performs row reduction step on A for row i.
    
    Parameters
    ----------
    A : np.array
        Matrix for factorisation.
    i : int
        Row index for reduction.
    n : int
        Size of matrix.
    pivot_value : float
        Pivot value for reduction.
        
    Returns
    -------
    A : np.array
        Matrix after row reduction.
    """
    # for each row BELOW the pivot
    for ii in range(i+1, n):
        multiplier = A[ii, i]/pivot_value	
        
        # OPTION 1
        # loop over values in row, make replacements 
        # **comment this for loop when making the OPTION 2 modification**

        # for j in range(i, n):	
        #     A[ii,j] = A[ii,j] - multiplier*A[i,j]   
            
        # **to do**
        # OPTION 2
        # FOR loops are SLOW in Python, can you vectorize the code
        # above? That is, use index slicing instead of a loop
        # **your code here**
        A[ii,i:n] = A[ii,i:n] - multiplier*A[i,i:n]
        
        A[ii,i] = multiplier                                        
    
    return A	

# **this function should be completed**
#                           ---------
def read_and_factor(fl):
    ''' **write docstring here**

        Notes:
        ------
        Reads and factorises a matrix by calling pre-written functions
        lu_read and lu_factor.
    '''
    # **delete the command below after writing your function**
    pass
    

#############################################################################
# Matrix multiplication functions
#############################################################################

# this function is complete 
def matmult(A,B):
    ''' Multiply two matrices together using THREE FOR LOOPS.

        Parameters
        ----------
        A : array-like
            Lefthand multiplying matrix.
        B : array-like
            Righthand multiplying matrix.

        Returns
        -------
        C : array-like
            Matrix product.

        Raises
        ------
        ValueError
            If inner dimensions of matrix product are inconsistent
    '''
    # check dimension consistency precondition
    if A.shape[1] != B.shape[0]:
        raise ValueError('Dimension inconsistency: A must have the same number of columns as B has rows.')
    
    # compute matrix product as dot products of rows and columns
    n = A.shape[1]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            sum = 0
            for k in range(n):
                sum += A[i,k]*B[k,j]
            C[i,j] = sum
    return C

# **this function should be modified**
#                           --------
def matmult2(A,B):
    ''' Multiply two matrices together using TWO FOR LOOPS and SLICING.

        Parameters
        ----------
        A : array-like
            Lefthand multiplying matrix.
        B : array-like
            Righthand multiplying matrix.

        Returns
        -------
        C : array-like
            Matrix product.

        Raises
        ------
        ValueError
            If inner dimensions of matrix product are inconsistent
    '''
    # check dimension consistency precondition
    if A.shape[1] != B.shape[0]:
        raise ValueError('Dimension inconsistency: A must have the same number of columns as B has rows.')
    
    # compute matrix product as dot products of rows and columns
    n = A.shape[1]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            # **replace this for loop using index slicing and the np.sum function**
            sum = 0
            for k in range(n):
                sum += A[i,k]*B[k,j]
            C[i,j] = sum
    return C

#############################################################################
# Other functions - these should not be modified
#############################################################################

def lu_read(filename):	
    ''' 
    Load cofficients of a linear system from a file.
    
    Parameters
    ----------
    filename : str
        Name of file containing A and b.
        
    Returns
    A : np.array
    -------
        Matrix for factorisation.
    b : np.array
        RHS vector.
        
    Notes
    -----
    filename should be point to a textfile 
    
    Examples
    --------
    The syntax for a determined linear system with four unknowns in the text file. 
    1x1 scalar for unknowns (row 0)
    4x4 matrix for coefficients (rows 1 to 4)
    1x4 matrix for RHS vector (row 5)

    4 
     2.0  3.0 -4.0  2.0
    -4.0 -5.0  6.0 -3.0
     2.0  2.0  1.0  0.0
    -6.0 -7.0 14.0 -4.0
     4.0 -8.0  9.0  6.0
    '''
    
    with open(filename, 'r') as fp:
        # Row dimension
        nums = fp.readline().strip()
        row = int(nums)
        
        A = []
        for i in range(row):
            nums = fp.readline().rstrip().split()
            A.append([float(num) for num in nums])
        A = np.array(A)
        
        b = []
        nums = fp.readline().rstrip().split()
        b.append([float(num) for num in nums])
        b = np.array(b)
        
    return A, b.T
def plot_scaling(x, y, fit_line=True, save=None):
    ''' plots x, y with the best fitted line
    '''
    f,ax = plt.subplots(1,1)                         # creates an empty figure
    ax.plot(x, y, 'bo')          			         # plots the results in log space
    p = np.polyfit(x, y, 1)    					# fits a line (first order polynomial)
    xlim = ax.get_xlim()                                        
    ax.set_xlim(xlim)                                # sets the axes limits
        # plot the fitted line and report the slope
    ax.plot(xlim, p[0]*np.array(xlim)+p[1], 'r-', label='slope = {:2.1f}'.format(p[0]))
    # tidy up and save the plot
    ax.set_xlabel('log(problem size)')
    ax.set_ylabel('log(execution time)')
    ax.set_title('time scaling of algorithm')
    ax.legend()
    try:
        md = {'a': os.environ['COMPUTERNAME']}
    except KeyError:
        try:
            import socket
            md = {'a': socket.gethostname()}
        except:
            md = None
            pass
    if save is None:
        plt.show()
    else: # removed papertype, frameon=None,
        plt.savefig(save, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', format=None, transparent=False, bbox_inches=None, pad_inches=0.1, metadata=md)
def square_matrix_rand(N):	
    ''' 
    Create a square matrix of normally distributed random numbers
    
    Parameters
    ----------
    N : integer
        Size of the square matrix (NxN)
        
    Returns
    -------
    A : np.array
        Square matrix 

    '''	
    return np.random.randn(N,N)
