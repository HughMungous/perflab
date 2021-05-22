# ENGSCI233: Lab - Performance
# perflab_practice.py

# PURPOSE:
# Practice exercises to investigate:
# - Profiling - which sections of your code are the most time consuming? 
# - Scaling - how does the time consumed scale with larger tasks?
# - Optimization - how can the we speed up the time consuming sections? 
# - Parallelization - how can we assign tasks to be completed simultaneously?

# PREPARATION:
# Notebook performance.ipynb.

# SUPPORTING MATERIAL (SP):
# SP1: cProfile library docs. https://docs.python.org/2/library/profile.html
# SP2: Multiprocessing Pool intro. https://www.youtube.com/watch?v=_1ZwkCY9wxk

# SUBMISSION:
# You should NOT SUBMIT this file.

# imports
from perflab_functions import*
import cProfile, pstats
from multiprocessing import Pool

from numpy.linalg import norm
from time import time
from copy import copy
   
# What does this 'if __name__ == "__main__":' mean?
# This allows you to combine script commands and functions in the same file. Function definitions
# sit outside the 'if __name__ == "__main__":' block, and script commands within it. This block is
# also important for avoiding recursion errors on Windows when using multiprocessing.
if __name__ == "__main__":

    ##############################################################################################
    # EXERCISE 1: Profiling using the cProfile library
    # In this section, we will check the time consumed by square_matrix_rand and matmult functions. 
    # We will use cProfile module to do this (See SP1). 
    
    # TO DO:
    #  1. SET the boolean below to True and RUN the code
    #  2. ANALYSE the profile printout
    #  3. what Python command is being profiled?
    #  4. what is the most (time) expensive function?
    #  5. INSPECT the docstring for matmult in perflab_functions.py
    #  6. MODIFY the code below to time the matmult function for A*A

    if False: 
        ## Profiling for square_matrix_rand 
        N = 200   # size of square matrix to be used
        print('\n Profiling for square_matrix_rand function \n ')

        # the run command creates a file called 'restats' that contains the timing information
        cProfile.run('A = square_matrix_rand(N)', 'restats')
        p = pstats.Stats('restats')
        p.sort_stats('time').print_stats(5)

        # **to do**
        A = np.random.rand(N,N)
        # write a command to create a random square matrix, 
        cProfile.run("matmult(A,A)", "westats")
        b = pstats.Stats("westats")
        b.sort_stats('time').print_stats(5)
        # write a command to multiply the matrix by itself, using matmult()
        # put the matrix multiplication command inside cProfile.run() and print the timing info.


    ##############################################################################################
    # EXERCISE 2: Scaling of Python matrix multiplication
    # In this section, we will study how the time consumed by matmult scales as we multiply 
    # larger and larger matrices. 
    
    # TO DO:
    # - RUN the code block below. Note the operation of the FOR loop and the plotting commands. 
    # - MODIFY the code to: 
    #     (1) create a square matrix A of size N 
    #     (2) multiply it by itself using matmult in perflab_functions.py
    #     (3) save the execution time for using the p.total_tt attribute
    # - INTERPRET the plot: is the scaling as you expect? Google 'time complexity matrix multiplication'
    #   and explain the result.

    if False:
        ts = []                                                # an empty list to store times
        Ns = [2**i for i in range(3,8)]                        # matrix sizes to test
        for N in Ns:                                           # loop over all problem sizes
            # ** complete the code below: **
            A = np.random.rand(N,N)
            cProfile.run("matmult(A,A)", "westats")

            b = pstats.Stats("westats")
            
            ts.append(b.total_tt)                            
        
        # plotting - these commands are complete
        plot_scaling(np.log10(Ns), np.log10(ts), fit_line=True, save='matmult_scaling.png')
       
    
    ##############################################################################################
    # EXERCISE 3: Optimising Python matrix multiplication
    # FOR loops in Python are slow. You should always be on the look out for opportunities to 
    # VECTORISE your code.
    # VECTORISE means to replace a loop that operates on an array one element at a time, with an array 
    # operation that performs the same operation in an elementwise fashion. 

    # TO DO:
    # - RUN the code below and verify that the three operations all produce the same result
    if False:
        # two vectors, of which we wish to take the dot product
        Arow = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])
        Bcol = np.array([100., 200., 300., 400., 500., 600., 700., 800., 900., 1000.])

        # OPTION 1: for loop
        sum = 0.
        for k in range(len(Arow)):
            sum += Arow[k]*Bcol[k]
        print('Option 1:', sum)

        # OPTION 2: vector operations
        ArowxBcol=Arow*Bcol             # element-wise multiplication of two vectors
        sum = np.sum(ArowxBcol)         # add the components of the resulting vector
        print('Option 2:', sum)

        # OPTION 3: Python np.dot()
        sum = np.dot(Arow, Bcol)        # Python's built-in function
        print('Option 3:', sum)
        
        
    # TO DO:
    # - MODIFY matmult2 in perflab_functions.py to replace the innermost for loop with a vector command
    # - QUANTIFY the speed up (to the nearest order of magnitude.)
    if False:
        N = 200   # size of square matrix to be used
        print('\n Profiling for matmult function \n ')
        A = square_matrix_rand(N)
        cProfile.run('matmult(A,A)', 'restats')
        print(pstats.Stats('restats').total_tt)

        # **your profiling code here**


    ##############################################################################################
    # EXERCISE 4: Parallelising matrix multiplication
    # **the code below is complete and is a demonstration only**
    # Suppose we want to multiply eight matrices [A, B,... H] together.
    #
    # In serial, the computation goes:
    # (0) AB = A x B; ABC = AB x C; ... ABCDEFGH = ABCDEFG x H
    #
    # Each calculation on line (0) depends on the previous. 
    # This is implemented in the timed FOR loop below.
    # SERIAL CALCULATION
    if True:
        N = 200                                                             # set matrix size
        A,B,C,D,E,F,G,H = [square_matrix_rand(N) for i in range(8)]         # generate matrices
        ABCDEFGHserial = copy(A)                                            
        t0 = time()
        for X in [B,C,D,E,F,G,H]:                                           # iteratively compute
            ABCDEFGHserial = matmult2(ABCDEFGHserial,X)                     # matrix product
        t1 = time()
        print('serial: {:3.2f} seconds'.format(t1-t0))                      # timing info.

    # Key parallelisation concepts: 
    # 	- Map: Divide work into multiple units 
    #	- Reduce: Agregate results back to a common single output. 
    #
    # Example of serial processing on two simultaneous tasks: 
    #	         --    --    --                  ...task 1
    #	        /  \  /  \  /  \
    #	>---- --    --    --    -- ---->>        ...task 2
    #   
    
    # Example of parallel processing on two simultaneous tasks: 
    #	       ------                            ...task 1
    #	      /      \
    #	>---- -------- ---->>                    ...task 2
    

    # We could also divide the computations into pairs:
    # (1) AB = A x B; CD = C x D; EF = E x F; GH = G x H;
    # (2) ABCD = AB x CD; EFGH = EF x GH;
    # Multiplications on line (1) do not depend on each other, they can be performed concurrently. 
    # The same is true for line (2).
    #
    # We then aggregate the results with the final computation:
    # (3) ABCDEFGH = ABCD x EFGH
    #
    # This parallel calculation is implemented and timed below
    # PARALLEL CALCULATION
    if False:
        ncpus = 2                                                    
        p = Pool(ncpus)                                 # create the multiprocessing pool 
        t0 = time()
        pairs = []
        for M1, M2 in zip([A,C,E,G],[B,D,F,H]):
            pairs.append([M1, M2])                      # create the input pairs (x4)
        outputs = p.starmap(matmult2, pairs)            # multiply pairs
        AB, CD, EF, GH = outputs                        # unpack the outputs (x4)
        pairs = []
        for M1, M2 in zip([AB,EF],[CD,GH]):
            pairs.append([M1, M2])                      # create the input pairs (x2)
        outputs = p.starmap(matmult2, pairs)            # multiply pairs
        ABCD, EFGH = outputs                            # unpack the outputs (x2)

        ABCDEFGHparallel = matmult2(ABCD,EFGH)                              # final mult.
        t1 = time()
        print('parallel: {:3.2f} seconds'.format(t1-t0))                    # timing info.

        # for this computation, you will need both this block and the one above set if True:
        print(norm(ABCDEFGHserial-ABCDEFGHparallel))                        # check identical

    # Run each of the code blocks above
    # - What is the parallel speedup for ncpus = 2? For ncpus = 4?
    # - Do the serial and parallel computations return the same answer? How do you know? If 
    #   they don't, why not?
    # - What does the 'zip' function above do?
    # **hint: run the code 
    #  a = np.arange(1,10); b = np.arange(11,20); print(a); print(b); print(list(zip(a,b)))
    # - the error norm between the SERIAL and PARALLEL increases with matrix size - why? 
