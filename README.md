# Ising_model

To print the output matrix uncomment print statements in .c or .cu code.
To select common rand function seed for comparisons of the algorithms comment srand(time(NULL)) in .c or .cu code.
To change threads per block change blocksize global variable in .c or .cu code.

Linux Execution:

./seq_ising N K > file.txt
#to print the output in a .txt file

./cuda_ising1 N K > file.txt
#to print the output in a .txt file

./cuda_ising2 N gridsize K > file.txt
#to print the output in a .txt file

./cuda_ising3 N gridsize K > file.txt
#to print the output in a .txt file

!Careful: gridsize * blocksize should be less or equal to N for the algorithms to work properly.Also, it is suggested that the parameters are powers of 2 to avoid unexpected results.

#Matlab files for the graphs are also uploaded as well as bash scripts used to run the programs in the Aristotelis structure.
