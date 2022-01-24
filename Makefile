CC=gcc
NVCC=nvcc
#CFLAGS=-O3

default: all

seq_ising:
	$(CC) -o seq_ising seq_ising.c	

cuda_ising1:
	$(NVCC) -o cuda_ising1 cuda_ising1.cu

cuda_ising2:
	$(NVCC) -o cuda_ising2 cuda_ising2.cu

cuda_ising3:
	$(NVCC) -o cuda_ising3 cuda_ising3.cu

cuda_ising3_old:
	$(NVCC) -o cuda_ising3_old cuda_ising3_old.cu


.PHONY: clean

all: seq_ising cuda_ising1 cuda_ising2 cuda_ising3 cuda_ising3_old

clean:
	rm -f seq_ising cuda_ising1 cuda_ising2 cuda_ising3 cuda_ising3_old seq_ising.out cuda_ising1.out cuda_ising2.out cuda_ising3.out cuda_ising3_old.out *~
