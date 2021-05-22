# ENGSCI233: Lab - Performance
# perflab_optimise.py

# PURPOSE:
# Assessed exercises for optimising the LU factorisation algorithm.

# PREPARATION:
# Work through the exercises in perflab_practice.py.

# SUBMISSION:
# - YOU MUST submit this file to complete the lab. 
# - DO NOT change the file name.
# - You should ALSO SUBMIT perflab_scaling.png from EXERCISE 1

# TO DO:
# - COMPLETE Exercises 1 through 4.

# imports
from perflab.perflab_functions import lu_factor
from perflab_functions import*
import cProfile, pstats
from glob import glob
from time import time
from multiprocessing import Pool
if __name__ == "__main__":
    
    ##############################################################################################
    # EXERCISE 1: Quantify scaling of lu_factor
    # In this section, you will study how lu_factor performs on larger and larger matrices. 
    # Note, do this exercise BEFORE optimising lu_factor (EXERCISE 3)
    #
    # TO DO:
    #  1. RUN lu_factor for square matrices of size 2^N where N ranges from 5 to 9 
    #  2. PLOT the scaling of lu_factor runtime in log-space
    #  3. SAVE your plot as perflab_scaling.png (use the plot_scaling() function from practice)
    #     (Do this BEFORE Exercise 2 below)
    #  4. ANSWER in perflab_questions.txt:
    #     WHAT is the scaling of LU factorisation in Big O notation?
    
    if False:        
        ts = []                                                # an empty list to store times
        Ns = [2**i for i in range(3,13)]                        # matrix sizes to test
        for N in Ns:                                           # loop over all problem sizes
            # ** complete the code below: **
            A = np.random.rand(N,N)
            cProfile.run("lu_factor(A)", "restats")

            b = pstats.Stats("restats")
            
            ts.append(b.total_tt)   

        plot_scaling(np.log10(Ns), np.log10(ts), fit_line=True, save='perflab_scaling.png')
        

    ##############################################################################################
    # EXERCISE 2: Profile lu_factor
    # In this section, you will identify the inefficiencies in lu_factor()
    #
    # TO DO:
    #  1. WRITE code to profile the function lu_factor for the square matrix below
    #  2. In perflab_questions.txt:
    #     EXPLAIN how the printout indicates that row_reduction() is the principle bottleneck

    if False: 
        N = 200
        A = square_matrix_rand(N)
        print('\n Profiling for lu_factor function \n ')  

        cProfile.run('lu_factor(A)', 'restats')
        p = pstats.Stats('restats')
        p.sort_stats('time').print_stats(5)

    
    ##############################################################################################
    # EXERCISE 3: Optimise lu_factor
    # In this section, you will fix the inefficiencies in lu_factor().  
    #
    # TO DO:
    #  1. OPTIMISE lu_factor() by vectorising the inner FOR loop in row_reduction()
    #  2. In perflab_questions.txt:
    #     QUANTIFY the speed-up of lu_factor() (to the nearest order of magnitude)

    if False: 
        N = 200
        A = square_matrix_rand(N)
        print('\n Profiling for lu_factor function \n ')  
        # **your profiling code here**
        cProfile.run('lu_factor(A)', 'restats')
        p = pstats.Stats('restats')
        p.sort_stats('time').print_stats(5)

    
    ##############################################################################################
    # EXERCISE 4: Parallel application of lu_factor
    # matrices.zip contains 1000 randomly sized matrices that need to be factorised
    # the code below reads the FIRST TWENTY matrices one at a time and factorises them 
    #
    # TO DO:
    #  1. EXTRACT the contents of 'matrices.zip' to the directory 'matrices'
    #  2. RUN the serial code below. In perflab_questions.txt, answer:
    #     HOW LONG does it take to factorise all the matrices? 
    #  3. PARALLELISE this factorisation task, making use of Pool, Pool.map, and 
    #     read_and_factor() (to be completed in perflab_functions.py). 
    #     In perflab_questions.txt, answer: What speedup can you achieve?
    #
    # NOTES:
    #  - DO NOT parallelise the lu_factor function
    #  - the code below makes use of 'glob' and 'np.genfromtxt' - you may wish to look up how these
    #    work, but you won't need to know how to use them until the Data module
    
    
    if True:
        # SERIAL calculation
        n = 20
        fls = glob('matrices/*.txt')[:n]            # 'glob' uses wildcards ('*') to detect files
        t0 = time()                                 #     conforming to a naming pattern
        for fl in fls:
            A = np.genfromtxt(fl,delimiter=',')     # read
            LU = lu_factor(A)                        # factor
        t1 = time()
        print('time to factorise {:d} matrices in SERIAL: {:2.1f} seconds'.format(n,t1-t0))

        # PARALLEL calculation
        ncpus = 8 # change for mac!!!!!
        for ncpu in range(2, ncpus+1):
            p = Pool(ncpu)
            t0 = time()
            p.map(lu_factor, [np.genfromtext(fl,delimter = ',') for fl in fls])
            t1 = time()
            print('factorising {:d} matrices with {:d} cpus: '.format(n,ncpu),t1-t0,' seconds')

