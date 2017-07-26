# PaperID3413
Critical Points Of An Autoencoder Can Provably Recover Sparsely Used Overcomplete Dictionaries

Datagen.m -> This code generates data for the  simulation.
simu5.m -> This code does the simulation.

To perform a simulation, first run Datagen.m code and then run simu5.m.

- choose a small dimension n= 100 say and a large dimension where the sparse code lives h = 256 say. 
- Now choose a random matrix A^* of dimension 100 x 256 and normalize each of its columns (l2-norm)
- Now measure the magnitude of the largest cross-column innerproduct for A^* and let this be equal to h^(-xi) 
  This defines your value of \xi. 
- Now as per theorem 3.2 we need p < min (1/3, \xi/2) 
  Choose a p accordingly but a sligtly larger p than this should be okay. 
 - Create lots of sparse vectors (x^*) of dimension h with have h^p non-zero elements.
- Crerate the corresponding y = A^*x^*. 
- Now initialize the deep net matrix W of dimension h times n with each of its column say a distance 1 or 2 away from the corresponding columns of A^*^T. Call this the "initial W" 
-  Now sample a bunch of ys and use this W to estimate the gradient as per the 3 equations in the last appendix. 
   The full gradient is a sum of the 3 derivatives coming from each of the 3 parts of loss function L. 
  For each i = 1 to h you will get one matrix i.e the gradient along that direction.  
-   Update the i-th row of W according to the ith component gradient obtained above.
- Keep repeating this till the gradient norm goes down close to 0. Put some threshold say 10^-6 or whatever works. 
- When the gradient is this low see whats the current value of W. 
 - Now normalize the columns of the W^T and this column-normalized-critical-W^T is your candidate A^*.
   Check if this each column of this column-normalized-critical-W^T is within a 1-ball of the corresponding column of A^*.    
   Choose and fix a set of test x^* vectors and see if on this the column-normalized-critical-W^T is acting like A^* or not.
   Is the average error on this test set of x^* smaller than what one started out with when evaluated with the intiial-W defined in step-7 above? 
- Plot this expected error on the test vectors as a function of p as you increase it and then vary say h too in more iterations of the experiment. 


n = 100
h = 256
mu = 2.998
zie = 0.2172
p = 0.1586
support, s = 2.4096

y = A_star * x_star
x_star belongs to dimension h
y belongs to dimension n

h^-zie = mu_by_root_n = max|<A_i, A_j>|
p < {zie/2,1/3}
|s| <= h^p
