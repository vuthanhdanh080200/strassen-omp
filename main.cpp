#include <omp.h>
#include <stdio.h>   
#include <stdlib.h>  
#include <fstream>
#include <iostream>

using namespace std;

/*#define THRESHOLD  32768*/ /* product size below which matmultleaf is used */  
int THRESHOLD;

double **createMatrix( int nRows, int nCols)
{
    //(step 1) allocate memory for array of elements of column
    double **ppi = new double*[nRows];

    //(step 2) allocate memory for array of elements of each row
    double *curPtr = new double[nRows * nCols];

    // Now point the pointers in the right place
    for( int i = 0; i < nRows; ++i)
    {
        *(ppi + i) = curPtr;
         curPtr += nCols;
    }
    return ppi;
}

void deleteMatrix(double** Array)
{
    delete [] *Array;
    delete [] Array;
}

void getBlock(double **X, int m, double **Y, int mf, int nf)
{
	for (int i = 0; i < m; i++) 
		X[i] = &Y[mf+i][nf];
}

void print(double** source, int m, int n){
    cout << "size " << m << " " << n << endl;
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            cout << source[i][j] << " ";
        }
        cout << endl;
    }
}

void add(double **T, int m, int n, double **X, double **Y)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T[i][j] = X[i][j] + Y[i][j];
}

void sub(double **T, int m, int n, double **X, double **Y)
{
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			T[i][j] = X[i][j] - Y[i][j];
}

void mulNaive(int m, int n, int p, double **A, double **B, double **C)    
{      

	for (int i = 0; i < m; i++)   
		for (int j = 0; j < n; j++) {
			C[i][j] = 0.0;  
			for (int k = 0; k < p; k++)   
				C[i][j] += A[i][k]*B[k][j];  
		} 
} 

void mul(int m, int n, int k, double** ma, double** mb, double** mc, int mode);

void strassen(int ml, int nl, int pl, double **A, double **B, double **C)
{
	if(((float)ml)*((float)nl)*((float)pl) <= THRESHOLD)   
		mulNaive(ml, pl, nl, A, B, C); 
	else {
		int m2 = ml/2;
		int n2 = nl/2;
		int p2 = pl/2;

		double **M1 = createMatrix(m2, n2);
		double **M2 = createMatrix(m2, n2);
		double **M3 = createMatrix(m2, n2);
		double **M4 = createMatrix(m2, n2);
		double **M5 = createMatrix(m2, n2);
		double **M6 = createMatrix(m2, n2);
		double **M7 = createMatrix(m2, n2);

		double **wAM1 = createMatrix(m2, p2);
		double **wBM1 = createMatrix(p2, n2);
		double **wAM2 = createMatrix(m2, p2);
		double **wBM3 = createMatrix(p2, n2);
		double **wBM4 = createMatrix(p2, n2);
		double **wAM5 = createMatrix(m2, p2);
		double **wAM6 = createMatrix(m2, p2);
		double **wBM6 = createMatrix(p2, n2);
		double **wAM7 = createMatrix(m2, p2);
		double **wBM7 = createMatrix(p2, n2);

		double **A11 = new double*[m2];
		double **A12 = new double*[m2];
		double **A21 = new double*[m2];
		double **A22 = new double*[m2];

		double **B11 = new double*[p2];
		double **B12 = new double*[p2];
		double **B21 = new double*[p2];
		double **B22 = new double*[p2];

		double **C11 = new double*[m2];
		double **C12 = new double*[m2];
		double **C21 = new double*[m2];
		double **C22 = new double*[m2];

		getBlock(A11, m2, A,  0,  0);
		getBlock(A12, m2, A,  0, p2);
		getBlock(A21, m2, A, m2,  0);
		getBlock(A22, m2, A, m2, p2);

		getBlock(B11, p2, B,  0,  0);
		getBlock(B12, p2, B,  0, n2);
		getBlock(B21, p2, B, p2,  0);
		getBlock(B22, p2, B, p2, n2);

		getBlock(C11, m2, C,  0,  0);
		getBlock(C12, m2, C,  0, n2);
		getBlock(C21, m2, C, m2,  0);
		getBlock(C22, m2, C, m2, n2);

        #pragma omp task
                {
            // M1 = (A11 + A22)*(B11 + B22)
                add(wAM1, m2, p2, A11, A22);
                add(wBM1, p2, n2, B11, B22);
                mul(m2, n2, p2, wAM1, wBM1, M1, 0);
                }

        #pragma omp task
                {
            //M2 = (A21 + A22)*B11
                add(wAM2, m2, p2, A21, A22);
                mul(m2, n2, p2, wAM2, B11, M2, 0);
                }

        #pragma omp task
                {
            //M3 = A11*(B12 - B22)
                sub(wBM3, p2, n2, B12, B22);
                mul(m2, n2, p2, A11, wBM3, M3, 0);
                }

        #pragma omp task
                {
            //M4 = A22*(B21 - B11)
                sub(wBM4, p2, n2, B21, B11);
                mul(m2, n2, p2, A22, wBM4, M4, 0);
                }

        #pragma omp task
                {
            //M5 = (A11 + A12)*B22
                add(wAM5, m2, p2, A11, A12);
                mul(m2, n2, p2, wAM5, B22, M5, 0);
                }

        #pragma omp task
                {
            //M6 = (A21 - A11)*(B11 + B12)
                sub(wAM6, m2, p2, A21, A11);
                add(wBM6, p2, n2, B11, B12);
                mul(m2, n2, p2, wAM6, wBM6, M6, 0);
                }

        #pragma omp task
                {
            //M7 = (A12 - A22)*(B21 + B22)
                sub(wAM7, m2, p2, A12, A22);
                add(wBM7, p2, n2, B21, B22);
                mul(m2, n2, p2, wAM7, wBM7, M7, 0);
                }
        #pragma omp taskwait

                for (int i = 0; i < m2; i++)
                    for (int j = 0; j < n2; j++) {
                        C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                        C12[i][j] = M3[i][j] + M5[i][j];
                        C21[i][j] = M2[i][j] + M4[i][j];
                        C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
                    }

		deleteMatrix(M1);
		deleteMatrix(M2);
		deleteMatrix(M3);
		deleteMatrix(M4);
		deleteMatrix(M5);
		deleteMatrix(M6);
		deleteMatrix(M7);

		deleteMatrix(wAM1);
		deleteMatrix(wBM1);
		deleteMatrix(wAM2);
		deleteMatrix(wBM3);
		deleteMatrix(wBM4);
		deleteMatrix(wAM5);
		deleteMatrix(wAM6);
		deleteMatrix(wBM6);
		deleteMatrix(wAM7);
		deleteMatrix(wBM7);

		delete[] A11; delete[] A12; delete[] A21; delete[] A22;
		delete[] B11; delete[] B12; delete[] B21; delete[] B22;
		delete[] C11; delete[] C12; delete[] C21; delete[] C22;
	}
}

void strassenWinograd(int m, int n, int k, double** ma, double** mb, double** mc){
    if(((float)m)*((float)k)*((float)n) <= THRESHOLD)   
		mulNaive(m, n, k, ma, mb, mc); 
	else {
    
        int m2 = m/2;
        int pp2 = k/2;
        int n2 = n/2;

        double **a11 = new double*[m2];
        double **a12 = new double*[m2];
        double **a21 = new double*[m2];
        double **a22 = new double*[m2];

        double **b11 = new double*[pp2];
        double **b12 = new double*[pp2];
        double **b21 = new double*[pp2];
        double **b22 = new double*[pp2];

        double **c11 = new double*[m2];
        double **c12 = new double*[m2];
        double **c21 = new double*[m2];
        double **c22 = new double*[m2];

        getBlock(a11, m2, ma,  0,  0);
        getBlock(a12, m2, ma,  0, pp2);
        getBlock(a21, m2, ma, m2,  0);
        getBlock(a22, m2, ma, m2, pp2);

        getBlock(b11, pp2, mb,  0,  0);
        getBlock(b12, pp2, mb,  0, n2);
        getBlock(b21, pp2, mb, pp2,  0);
        getBlock(b22, pp2, mb, pp2, n2);

        getBlock(c11, m2, mc,  0,  0);
        getBlock(c12, m2, mc,  0, n2);
        getBlock(c21, m2, mc, m2,  0);
        getBlock(c22, m2, mc, m2, n2);

        double** s1 = createMatrix(m2, pp2);
        double** s2 = createMatrix(m2, pp2);
        double** s3 = createMatrix(m2, pp2);
        double** s4 = createMatrix(m2, pp2);

        double** t1 = createMatrix(pp2, n2);
        double** t2 = createMatrix(pp2, n2);
        double** t3 = createMatrix(pp2, n2);
        double** t4 = createMatrix(pp2, n2);

        double** p1 = createMatrix(m2, n2);
        double** p2 = createMatrix(m2, n2);
        double** p3 = createMatrix(m2, n2);
        double** p4 = createMatrix(m2, n2);
        double** p5 = createMatrix(m2, n2);
        double** p6 = createMatrix(m2, n2);
        double** p7 = createMatrix(m2, n2);
        
        #pragma omp task
        {
            //S1 = A21 + A22
            add(s1, m2, pp2, a21, a22);
            //add1(a21, a22, s1, m2, pp2);
            //S2 = S1 - A11
            sub(s2, m2, pp2, s1, a11);
            //sub1(s1, a11, s2, m2, pp2);
            //S3 = A11 - A21
            sub(s3, m2, pp2, a11, a21);
            //sub1(a11, a21, s3, m2, pp2);
            //S4 = A12 - S2
            sub(s4, m2, pp2, a12, s2);
            //sub1(a12, s2, s4, m2, pp2);
        }
        
        #pragma omp task
        {
            //T1 = B12 - B11
            sub(t1, pp2, n2, b12, b11);
            //sub1(b12, b11, t1, pp2, n2);
            //T2 = B22 - T1
            sub(t2, pp2, n2, b22, t1);
            //sub1(b22, t1, t2, pp2, n2);
            //T3 = B22 - B12
            sub(t3, pp2, n2, b22, b12);
            //sub1(b22, b12, t3, pp2, n2);
            //T4 = B21 - T2
            sub(t4, pp2, n2, b21, t2);
            //sub1(b21, t2, t4, pp2, n2);
        }

        #pragma omp taskwait
        
        #pragma omp task
        {
            //P1 = A11*B11
            mul(m2, n2, pp2, a11, b11, p1, 1);
        }

        #pragma omp task
        {
            //P2 = A12*B21
            mul(m2, n2, pp2, a12, b21, p2, 1);
        }

        #pragma omp task
        {
            //P3 = S1*T1
            mul(m2, n2, pp2, s1, t1, p3, 1);
        }

        #pragma omp task
        {
            //P4 = S2*T2
            mul(m2, n2, pp2, s2, t2, p4, 1);
        }

        #pragma omp task
        {
            //P5 = S3*T3
            mul(m2, n2, pp2, s3, t3, p5, 1);
        }

        #pragma omp task
        {
            //P6 = S4*B22
            mul(m2, n2, pp2, s4, b22, p6, 1);
        }

        #pragma omp task
        {
            //P7 = A22*T4
            mul(m2, n2, pp2, a22, t4, p7, 1);
        }
        
        #pragma omp taskwait
        
        //C11 = U1 = P1 + P2

        //U2 = P1 + P4

        //U3 = U2 + P5 = P1 + P4 + P5

        //C21 = U4 = U3 + P7 = P1 + P4 + P5 + P7

        //C22 = U5 = U3 + P3 = P1 + P4 + P5 + P3
        
        //U6 = U2 + P3 = P1 + P4 + P3
        
        //C12 = U7 = U6 + P6 = P1 + P4 + P3 + P6
        for(int i=0; i<m2; i++){
            for(int j=0; j<n2; j++){
                c11[i][j] = p1[i][j] + p2[i][j];
                c12[i][j] = p1[i][j] + p4[i][j] + p3[i][j] + p6[i][j];
                c21[i][j] = p1[i][j] + p4[i][j] + p5[i][j] + p7[i][j];
                c22[i][j] = p1[i][j] + p4[i][j] + p3[i][j] + p5[i][j];
            }
        }
       
        deleteMatrix(s1);
        deleteMatrix(s2);
        deleteMatrix(s3);
        deleteMatrix(s4);

        deleteMatrix(t1);
        deleteMatrix(t2);
        deleteMatrix(t3);  
        deleteMatrix(t4);

        deleteMatrix(p1);
        deleteMatrix(p2);
        deleteMatrix(p3);
        deleteMatrix(p4);
        deleteMatrix(p5);
        deleteMatrix(p6);
        deleteMatrix(p7);

        delete[] a11, delete[] a12, delete[] a21, delete[] a22;
        delete[] b11, delete[] b12, delete[] b21, delete[] b22;
        delete[] c11, delete[] c12, delete[] c21, delete[] c22;
    }
}

void mul(int m, int n, int k, double** ma, double** mb, double** mc, int mode){
    // using dynamic peeling when matrix size is odd
    if(((double)m)*((double)k)*((double)n) <= THRESHOLD){
        mulNaive(m, n, k, ma, mb, mc); 
    }
    else if((m%2 == 0) && (k%2 == 0) && (n%2 == 0)){
        if(mode == 0){
            strassen(m, n, k, ma, mb, mc);
        }
        else if(mode == 1){
            strassenWinograd(m, n, k, ma, mb, mc);
        }
    }
    else {
        int mBlock = m-1;
        int kBlock = k-1;
        int nBlock = n-1;

        double **a11 = new double*[mBlock];
		double **a12 = new double*[mBlock];
		double **a21 = new double*[1];
		double **a22 = new double*[1];

		double **b11 = new double*[kBlock];
		double **b12 = new double*[kBlock];
		double **b21 = new double*[1];
		double **b22 = new double*[1];

		double **c11 = new double*[mBlock];
		double **c12 = new double*[mBlock];
		double **c21 = new double*[1];
		double **c22 = new double*[1];

		getBlock(a11, mBlock, ma,  0,  0);
		getBlock(a12, mBlock, ma,  0, kBlock);
		getBlock(a21, 1, ma, mBlock,  0);
		getBlock(a22, 1, ma, mBlock, kBlock);

		getBlock(b11, kBlock, mb,  0,  0);
		getBlock(b12, kBlock, mb,  0, nBlock);
		getBlock(b21, 1, mb, kBlock,  0);
		getBlock(b22, 1, mb, kBlock, nBlock);

        getBlock(c11, mBlock, mc,  0,  0);
		getBlock(c12, mBlock, mc,  0, nBlock);
		getBlock(c21, 1, mc, mBlock,  0);
		getBlock(c22, 1, mc, mBlock, nBlock);

        //C11 = A11*B11 + a12*b21
        double** temp1 = createMatrix(mBlock, nBlock);
        double** temp2 = createMatrix(mBlock, nBlock);
        //c12 = A11*b12 + a12*b22
        double** temp3 = createMatrix(mBlock, 1);
        double** temp4 = createMatrix(mBlock, 1);
        //c21 = a21*B11 + a22*b21
        double** temp5 = createMatrix(1, nBlock);
        double** temp6 = createMatrix(1, nBlock);
        //c22 = a21*b12 + a22*b22
        double** temp7 = createMatrix(1, 1);
        double** temp8 = createMatrix(1, 1);

        mul(mBlock, nBlock, kBlock, a11, b11, temp1, mode);
        mulNaive(mBlock, nBlock, 1, a12, b21, temp2);
        add(c11, mBlock, nBlock, temp1, temp2);

        mulNaive(mBlock, 1, kBlock, a11, b12, temp3);
        mulNaive(mBlock, 1, 1, a12, b22, temp4);
        add(c12, mBlock, 1, temp3, temp4);

        mulNaive(1, nBlock, kBlock, a21, b11, temp5);
        mulNaive(1, nBlock, 1, a22, b21, temp6);
        add(c21, 1, nBlock, temp5, temp6);

        mulNaive(1, 1, kBlock, a21, b12, temp7);
        mulNaive(1, 1, 1, a22, b22, temp8);
        add(c22, 1, 1, temp7, temp8);
        
        delete[] a11, delete[] a12, delete[] a21, delete[] a22;
        delete[] b11, delete[] b12, delete[] b21, delete[] b22;
        delete[] c11, delete[] c12, delete[] c21, delete[] c22;

        deleteMatrix(temp1);
        deleteMatrix(temp2);
        deleteMatrix(temp3);
        deleteMatrix(temp4);

        deleteMatrix(temp5);
        deleteMatrix(temp6);
        deleteMatrix(temp7);  
        deleteMatrix(temp8);
    }

}

void matmultS(int m, int n, int p, double **A, double **B, double **C, int mode)
{   
#pragma omp parallel 
  {
#pragma omp single
	{
    // print(C, m, n);
    mul(m, n, p, A, B, C, mode);
	}
  }
} 

  
int main(int argc, char* argv[])   
{      

    int rawSize = atoi(argv[1]);
    THRESHOLD = atoi(argv[2]);
    int proc = atoi(argv[3]);
    int mode = atoi(argv[4]);

    double start, end;

    double **rawA = createMatrix(rawSize, rawSize);
    double **rawB = createMatrix(rawSize, rawSize);
    int i, j;   
        for (int i = 0; i < rawSize; i++){
            for (int j = 0; j < rawSize; j++){
                rawA[i][j] = (double)i/(double)rawSize;
                rawB[i][j] = (double)j/(double)rawSize;
            }
        }
    cout << "--------------------------------------------" << endl;
    double **C = createMatrix(rawSize, rawSize);

	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(proc);

	double time = omp_get_wtime();

  	matmultS(rawSize, rawSize, rawSize, rawA, rawB, C, mode);

  	time = omp_get_wtime() - time;

	printf("Checking the results...\n");
	double norm = 0.0;
	for(int i=0; i<rawSize; i++){
		for(int j=0; j<rawSize; j++){
			norm += (C[i][j]-(double)(i*j)/(double)rawSize)*(C[i][j]-(double)(i*j)/(double)rawSize);
		}
	}
	if(norm > 1e-10)
		printf("Something is wrong... Norm is equal to %f\n", norm);
	else    
		printf("Yup, we're good!\n");
	
	printf("Computing time: %f\n", time); 

  return 0;   
}  
