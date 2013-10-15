/*
 ============================================================================
 Name        : lab3.c
 Author      : Daniil Stepenskiy
 Version     :
 Copyright   : 
 Description : Lab3
 ============================================================================
 */
#include <math.h> 
#include <mpi.h>
#include <iostream>
#include <omp.h>
using namespace std;
 
int main(int argc, char *argv[]) {
	int n, rank, size, i;

	MPI::Init(argc, argv);
	size = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();

	int startTime = MPI::Wtime();
	int endTime = NULL;



	MPI::Finalize();
	return 0;
}

