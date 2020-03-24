# Number of global topics
int32 K_global
# Robot identifiers
string[] robots
# Ks[i] = number of topics known by i'th robot
int32[] Ks
# matches[\sum_{i=0}^n Ks[i] + j] = which global topic the j'th topic of the n'th robot is associated with
int32[] matches