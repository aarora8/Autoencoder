# PaperID3413
Critical Points Of An Autoencoder Can Provably Recover Sparsely Used Overcomplete Dictionaries

Datagen.m -> This code generates data for the  simulation.
simu5.m -> This code does the simulation.

To perform an simulation, first run Datagen.m coe and then run simu5.m.


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
