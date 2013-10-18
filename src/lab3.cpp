/*
 ============================================================================
 Name        : lab3.cpp
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
#include <cstdlib>

#include <sstream>
#include <string>
#include <exception>
#include <fstream>
using namespace std;

const double deltaT = 1.0 / 24.0;
const char csvDelimiter = ';';

//#define DEBUG_DISPATCHING
//#define DEBUG_GHOST_SWAP
#define DEBUG_TIME

#define MASTER_NODE 0

#define NO_NEIGHBOUR -1
#define MESSAGE_BOTTOM_LINE 1
#define MESSAGE_TOP_LINE 2

template <typename T>
T readValue(const std::string &s) {
	std::stringstream ss(s);
	T res;
	ss >> res;
	if (ss.fail()) {
		throw std::string("Failed to read from" + s);
	}
	return res;
}

double initial(int x, int y) {
	if ((x-10.0)*(x-10.0) + (y-10.0)*(y-10.0) < 10 * 10)
		return -10*sqrt(10.0*10.0 - (x-10.0)*(x-10.0) - (y-10.0)*(y-10.0));
	return 0;
}

double boundary_l(int y, double t) { // f(0, y, t)
	return 0;
}

double boundary_r(int y, double t) { // f(X_SIZE - 1, y, t)
	return 0;
}

double boundary_b(int x, double t) { // f(x, 0, t)
	return 0;
}

double f(int x, int y, double t) {
	//
	//return sin(2 * M_PI * x / 10) * sin(2 * M_PI * y / 10) * t;
	if ((x >= 6 && x <= 14 && y >= 4 && y<= 6) || (x >=6 && x <=7 && y >=4 && y <= 12 ) ||
			(x >=13 && x <=14 && y >=4 && y <= 12 )) {
		return t;
	}
	return 0;
}

double boundary_t(int x, double t) { // f(x, Y_SIZE - 1, t)
	return 0;
}
 
double** allocate(int sizeX, int sizeY) {
	double** result;

	result = new double*[sizeX];
	for (int i = 0; i < sizeX; ++i) {
		result[i] = new double[sizeY];
	}

	return result;
}

void destroy(double** matrix, int sizeX, int sizeY) {
	for (int i = 0; i < sizeX; ++i) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

void printMatrixWithT(ostream &stream, double** u, int sizeX, int sizeY, int offsetX, double t) {
	for (int x = 0; x < sizeX; ++x) {
		for (int y = 0; y < sizeY; ++y) {
			stream << t << csvDelimiter
			       << x + offsetX << csvDelimiter
			       << y << csvDelimiter
			       << u[x][y] << endl;
		}
	}
}

void printMatrixWithT(FILE* f, double** u, int sizeX, int sizeY, int offsetX, double t) {
	for (int x = 0; x < sizeX; ++x) {
		for (int y = 0; y < sizeY; ++y) {
			fprintf(f, "%f%c%d%c%d%c%f\n", t, csvDelimiter, x + offsetX, csvDelimiter, y, csvDelimiter,
					u[x][y]);
			/*stream << t << csvDelimiter
			       << x + offsetX << csvDelimiter
			       << y << csvDelimiter
			       << u[x][y] << endl;*/
		}
	}
}

void swapGhostCols(double** uOld, double* uGhostLeft, double* uGhostRight, int sliceSizeX, int sliceSizeY,
		int neighbourLeft, int neighbourRight, int nodeId, double t) {
	// At first even swaps right with odd (or initializes with boundaries if no neighbour)
	// Secondly even swaps left with odd (or ...)
	if (nodeId % 2 == 0) {
		if (neighbourRight != NO_NEIGHBOUR) {
			MPI::COMM_WORLD.Sendrecv(uOld[sliceSizeX - 1], sliceSizeY, MPI::DOUBLE,
					neighbourRight, MESSAGE_BOTTOM_LINE,
					uGhostRight, sliceSizeY, MPI::DOUBLE, neighbourRight, MESSAGE_TOP_LINE);
		} else {
			#pragma omp parallel for shared(uGhostRight, t)
			for (int y = 0; y < sliceSizeY; ++y) {
				uGhostRight[y] = boundary_r(y, t);
			}
		}
		if (neighbourLeft != NO_NEIGHBOUR) {
			MPI::COMM_WORLD.Sendrecv(uOld[0], sliceSizeY, MPI::DOUBLE,
					neighbourLeft, MESSAGE_TOP_LINE,
					uGhostLeft, sliceSizeY, MPI::DOUBLE, neighbourLeft, MESSAGE_BOTTOM_LINE);
		} else {
			#pragma omp parallel for shared(uGhostRight, t)
			for (int y = 0; y < sliceSizeY; ++y) {
				uGhostLeft[y] = boundary_l(y, t);
			}
		}
	} else /* nodeId % 2 != 0 */ {
		if (neighbourLeft != NO_NEIGHBOUR) {
			MPI::COMM_WORLD.Sendrecv(uOld[0], sliceSizeY, MPI::DOUBLE,
					neighbourLeft, MESSAGE_TOP_LINE,
					uGhostLeft, sliceSizeY, MPI::DOUBLE, neighbourLeft, MESSAGE_BOTTOM_LINE);
		} else {
			#pragma omp parallel for shared(uGhostRight, t)
			for (int y = 0; y < sliceSizeY; ++y) {
				uGhostLeft[y] = boundary_l(y, t);
			}
		}
		if (neighbourRight != NO_NEIGHBOUR) {
			MPI::COMM_WORLD.Sendrecv(uOld[sliceSizeX - 1], sliceSizeY, MPI::DOUBLE,
					neighbourRight, MESSAGE_BOTTOM_LINE,
					uGhostRight, sliceSizeY, MPI::DOUBLE, neighbourRight, MESSAGE_TOP_LINE);
		} else {
			#pragma omp parallel for shared(uGhostRight, t)
			for (int y = 0; y < sliceSizeY; ++y) {
				uGhostRight[y] = boundary_r(y, t);
			}
		}
	}
}

void calculateLayer(double** uOld, double** uNew,
		double* uGhostLeft, double* uGhostRight,
		int sliceSizeX, int sliceSizeY, int offsetX,
		double t, double deltaT) {
	// boundaries
	#pragma omp parallel for
	for (int x = 0; x < sliceSizeX; ++x) {
		uNew[x][0] = boundary_t(x + offsetX, t);
		uNew[x][sliceSizeY - 1] = boundary_b(x + offsetX, t);
	}
	// left and right colons
	#pragma omp parallel for
	for (int y = 1; y < sliceSizeY + 1; ++y) {
		uNew[0][y] = uOld[0][y] +
				deltaT * (uOld[1][y] + uGhostLeft[y] - 2 * uOld[0][y]) +
				deltaT * (uOld[0][y + 1] + uOld[0][y - 1] - 2 * uOld[0][y]) +
				deltaT * f(0, y, t);
		uNew[sliceSizeX - 1][y] = uOld[sliceSizeX - 1][y] +
				deltaT * (uOld[sliceSizeX - 2][y] + uGhostRight[y] - 2 * uOld[sliceSizeX - 1][y]) +
				deltaT * (uOld[sliceSizeX - 1][y + 1] +
						uOld[sliceSizeX - 1][y - 1] - 2 * uOld[sliceSizeX - 1][y]) +
						deltaT * f(sliceSizeX - 1, y, t);
	}
	// golden mean
	#pragma omp parallel for
	for (int x = 1; x < sliceSizeX - 2; ++x) {
		for (int y = 1; y < sliceSizeY - 2; ++y) {
			uNew[x][y] = uOld[x][y] +
					deltaT * (uOld[x + 1][y] + uOld[x - 1][y] - 2 * uOld[x][y]) +
					deltaT * (uOld[x][y + 1] + uOld[x][y - 1] - 2 * uOld[x][y]) +
					deltaT * f(x, y, t);
		}
	}
}

void swapMatrices(double** uOld, double** uNew, int sizeX, int sizeY) {

	for (int x = 0; x < sizeX; ++x) {
		#pragma omp parallel for
		for (int y = 0; y < sizeY; ++y) {
			swap(uOld[x][y], uNew[x][y]);
		}
	}
}

int main(int argc, char *argv[]) {
	int nodeId, nodeCount;
	int ompCount, sizeX, sizeY, sizeT;
	int workerCount;

	MPI::Init(argc, argv);
	nodeCount = MPI::COMM_WORLD.Get_size();
	workerCount = nodeCount;
	nodeId = MPI::COMM_WORLD.Get_rank();

	if (argc != 5) return 1;

	double startTime = MPI::Wtime();

	ompCount = readValue<int>(argv[1]);
	omp_set_dynamic(0);
	omp_set_num_threads(ompCount);
	sizeX = readValue<int>(argv[2]);
	sizeY = readValue<int>(argv[3]);
	sizeT = readValue<int>(argv[4]);

	if (workerCount > sizeX) {
		workerCount = sizeX;
	}
	if (nodeId >= workerCount) {
		return 0;
	}

	int colCount = sizeX / workerCount;
	int colFrom = nodeId * colCount;
	int colTo = (nodeId + 1) * colCount;
	if (sizeX / workerCount != 0 && nodeId < sizeX % workerCount) {
		++colCount;
		if (nodeId != 0) {
			colFrom += nodeId;
			colTo += nodeId + 1;
		} else {
			++colTo;
		}

	} else {
		colFrom += sizeX % workerCount;
		colTo += sizeX % workerCount;
	}
	int sliceSizeX = colTo - colFrom;
	int sliceSizeY = sizeY;
	int offsetX = colFrom;

	#ifdef DEBUG_DISPATCHING
		cout << nodeId << ": [" << colFrom << ", " << colTo << ") " << sliceSizeX << endl;
		MPI::COMM_WORLD.Barrier();
	#endif

	int neighbourLeft = (nodeId <= 0) ? NO_NEIGHBOUR : nodeId - 1;
	int neighbourRight = (nodeId + 1 >= workerCount) ? NO_NEIGHBOUR : nodeId + 1;
	#ifdef DEBUG_DISPATCHING
		cout << nodeId << ": t " << neighbourLeft << ", b " << neighbourRight << endl;
		MPI::COMM_WORLD.Barrier();
	#endif


	double** uOld; double** uNew;
	uOld = allocate(sliceSizeX, sliceSizeY);
	uNew = allocate(sliceSizeX, sliceSizeY);
	double* uGhostLeft = new double[sliceSizeY];
	double* uGhostRight = new double[sliceSizeY];

	stringstream nodeIdSs;
	nodeIdSs << "node" << nodeId << ".csv";
	FILE* f = fopen(nodeIdSs.str().c_str(),"w");
	#pragma omp parallel for
	for (int x = 0; x < sliceSizeX; ++x) {
		for (int y = 0; y < sliceSizeY; ++y) {
			uOld[x][y] = initial(x + offsetX, y);
		}
	}
	printMatrixWithT(f, uOld, sliceSizeX, sliceSizeY, offsetX, 0);

	for (int t = 0; t < sizeT; ++t) {
		swapGhostCols(uOld, uGhostLeft, uGhostRight, sliceSizeX, sliceSizeY,
				neighbourLeft, neighbourRight, nodeId, static_cast<double>(t) * deltaT);
		calculateLayer(uOld, uNew, uGhostLeft, uGhostRight,
				sliceSizeX, sliceSizeY, offsetX, static_cast<double>(t) * deltaT, deltaT);
		//printMatrixWithT(dataStream, uNew, sliceSizeX, sliceSizeY, offsetX,
		//		static_cast<double>(t) * deltaT);
		printMatrixWithT(f, uNew, sliceSizeX, sliceSizeY, offsetX,
						static_cast<double>(t) * deltaT);
		swapMatrices(uOld, uNew, sliceSizeX, sliceSizeY);
		#ifdef DEBUG_GHOST_SWAP
			for (int i = 0; i < sliceSizeY; ++i) {
				dataStream << "(l " << uGhostLeft[i] << ", r " << uGhostRight[i] << ")" << endl;
			}
			MPI::COMM_WORLD.Barrier();
		#endif
	}


	fclose(f);
	//dataStream.close();
	delete[] uGhostLeft;
	delete[] uGhostRight;
	destroy(uOld, sliceSizeX, sliceSizeY);
	destroy(uNew, sliceSizeX, sliceSizeY);
	double endTime = MPI::Wtime();

	MPI::Finalize();

	#ifdef DEBUG_TIME
		if (nodeId == MASTER_NODE) {
			cout << endTime - startTime << endl;
		}
	#endif

	return 0;
}

