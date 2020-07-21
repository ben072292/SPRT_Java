//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example computes real matrix C=alpha*A*B+beta*C using Intel(R) MKL
*   function dgemm, where A, B, and C are matrices and alpha and beta are
*   scalars in double precision.
*
*   In this simple example, practices such as memory management, data alignment,
*   and I/O that are necessary for good programming style and high MKL
*   performance are omitted to improve readability.
********************************************************************************/

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.mkl.global.mkl_rt.*;

/**
 * Example code for testing JNI Intel MKL
 * @author Ben
 *
 */
public class DGEMMExample {
    public static void main(String[] args) throws Exception {
    	long startTime = System.nanoTime();
    	long endTime;
    	long timeElapsed;
        System.out.println("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
                         + " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
                         + " alpha and beta are double precision scalars\n");

        int m = 10000, p = 10000, n = 10000;
        System.out.printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
                        + " A(%dx%d) and matrix B(%dx%d)\n\n", m, p, p, n);
        double alpha = 1.0, beta = 0.0;

        System.out.println(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
                         + " performance \n");
        DoublePointer A = new DoublePointer(MKL_malloc(m * p * Double.BYTES, 64));
        DoublePointer B = new DoublePointer(MKL_malloc(p * n * Double.BYTES, 64));
        DoublePointer C = new DoublePointer(MKL_malloc(m * n * Double.BYTES, 64));
        if (A.isNull() || B.isNull() || C.isNull()) {
            System.out.println( "\n ERROR: Can't allocate memory for matrices. Aborting... \n");
            MKL_free(A);
            MKL_free(B);
            MKL_free(C);
            System.exit(1);
        }

        System.out.println(" Intializing matrix data \n");
        DoubleIndexer Aidx = DoubleIndexer.create(A.capacity(m * p));
        for (int i = 0; i < m * p; i++) {
            A.put(i, (double)(i + 1));
        }

        DoubleIndexer Bidx = DoubleIndexer.create(B.capacity(p * n));
        for (int i = 0; i < p * n; i++) {
            B.put(i, (double)(-i - 1));
        }

        DoubleIndexer Cidx = DoubleIndexer.create(C.capacity(m * n));
        for (int i = 0; i < m * n; i++) {
            C.put(i, 0.0);
        }
        
        endTime = System.nanoTime();
        timeElapsed = endTime - startTime;
        System.out.println("1: Execution time in milliseconds: " + timeElapsed / 1e6);
        
        startTime = System.nanoTime();
        System.out.println(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, p, alpha, A, p, B, n, beta, C, n);
        System.out.println("\n Computations completed.\n");
        
        endTime = System.nanoTime();
        timeElapsed = endTime - startTime;
        System.out.println("2: Execution time in milliseconds: " + timeElapsed / 1e6);
        
        startTime = System.nanoTime();
        System.out.println("\n Deallocating memory \n");
        MKL_free(A);
        MKL_free(B);
        MKL_free(C);

        System.out.println(" Example completed. \n");
        endTime = System.nanoTime();
        timeElapsed = endTime - startTime;
        System.out.println("3: Execution time in milliseconds: " + timeElapsed / 1e6);
        
        
        
        
        
        startTime = System.nanoTime();
        

        System.out.println(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
                         + " performance \n");
        

        System.out.println(" Intializing matrix data \n");
        double[] AA = new double[m *p];
        for (int i = 0; i < m * p; i++) {
            AA[i] = i+1;
        }
        
        double[] BB = new double[p * n];
        for (int i = 0; i < m * p; i++) {
            BB[i] = -i-1;
        }
        
        double[] CC = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            CC[i] = 0;
        }
        
        endTime = System.nanoTime();
        timeElapsed = endTime - startTime;
        System.out.println("1: Execution time in milliseconds: " + timeElapsed / 1e6);
        
        startTime = System.nanoTime();
        System.out.println(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, p, alpha, AA, p, BB, n, beta, CC, n);
        System.out.println("\n Computations completed.\n");
        
        endTime = System.nanoTime();
        timeElapsed = endTime - startTime;
        System.out.println("2: Execution time in milliseconds: " + timeElapsed / 1e6);
        
        
        System.exit(0);
    }
}

/* Sample Output

This example computes real matrix C=alpha*A*B+beta*C using 
Intel(R) MKL function dgemm, where A, B, and  C are matrices and 
alpha and beta are double precision scalars

Initializing data for matrix multiplication C=A*B for matrix 
A(10000x10000) and matrix B(10000x10000)

Allocating memory for matrices aligned on 64-byte boundary for better 
performance 

Intializing matrix data 

1: Execution time in milliseconds: 7266.3567
Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface 


Computations completed.

2: Execution time in milliseconds: 31347.3849

Deallocating memory 

Example completed. 

3: Execution time in milliseconds: 98.9658
Allocating memory for matrices aligned on 64-byte boundary for better 
performance 

Intializing matrix data 

1: Execution time in milliseconds: 2821.9783
Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface 


Computations completed.

2: Execution time in milliseconds: 33363.7482

*/