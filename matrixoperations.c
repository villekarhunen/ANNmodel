#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

/*

Package for matrix operations. 

Matrix datastructure consists of information about dimensions, matrixarray and permanency parameter.

Parameter permanency describes how dynamically created matrices behave. By setting it to 1, one 
ensures that matrix won't be removed from memory unless programmer wants that.
By setting it 0, function Cdelete (conditional delete) deletes it from memory permanently.

Cdelete(matrix) are called in most functions for each input parameter to delete every temporary matrix
immediately.

By using temporary matrices, one can stack commands:
For example adding operation add(A, t(A)) would lead to memory leak, if the transpose matrix t(A) would not be permanent since
it's address would be lost in function call.

Many operations have two almost same implementations: add and addP -> P indicates that return matrix is permanent.

Permanent matrices must be deleted by deleteM(matrix).

Package includes some basic operations such as add, prod, transpose and also linear system solvers by gaussian elimination and LU-decomposition.

*/

struct vector {
    int size;
    float* vectorArray;
    int permanency;
};

struct matrix {
    int ncol;
    int nrow;
    float **matrArray;
    int permanency;
};

//Delete given matrix
int deleteM(struct matrix* m) {
    if (m==NULL) {return 1;}
    for (int i = 0; i<m->nrow; i++) {
        free(m->matrArray[i]);
    }
    free(m->matrArray);
    free(m);
    return 1;
}
float elem(struct matrix *m, int row, int col) {
    return m->matrArray[row][col];
}

struct matrix* setElem(struct matrix *m, int row, int col, float newElem) {
    m->matrArray[row][col] = newElem;
    return m;
}

int deleteV(struct vector* v) {
    free(v->vectorArray);
    free(v);
    return 1;
}

// Delete matrix conditionally
int Cdelete(struct matrix* m) {
    return (m->permanency==0) ? deleteM(m) : 0;
}
int CdeleteV(struct vector* v) {
    return (v->permanency==0) ? deleteV(v) : 0;
}

struct vector* newVecD(int permanency, int size, ...) {
    struct vector* newvector = malloc(sizeof(struct vector));
    newvector->vectorArray = malloc(sizeof(float)*size);

    va_list args;
    va_start(args, size);
    double a;
    for (int index=0; index<size; index++) {
        a = va_arg(args, double);
        newvector->vectorArray[index] = (float)a;
    }
    va_end(args);
    newvector->size = size;
    newvector->permanency = permanency;
    return newvector;
}

struct matrix* emptyMatrD(int rows, int cols, int permanency) {
    /*
    Creates new matrix with empty content.
    permanency 1 -> Matrix must be manually deleted from dynamic memory if not used in another functions before ending
    permanency 0 -> Matrix will be deleted from dynamic memory after it has been used by another function.
    */
    struct matrix *nm = malloc(sizeof(struct matrix));
    nm->ncol = cols;
    nm->nrow = rows;
    nm->permanency = permanency;
    nm->matrArray = malloc(rows * sizeof(float*));
    for (int row = 0; row < rows; row++) {
        nm->matrArray[row] = calloc(cols, sizeof(float));
        if (!nm->matrArray[row]) {
            perror("Failed to allocate row data");
            // Free previously allocated rows before returning
            for (int j = 0; j < row; j++) {
                free(nm->matrArray[j]);
            }
            free(nm->matrArray);
            free(nm);
            return NULL;
        }
    }
    return nm;
}

struct matrix* I(int n, int permanency) {
    struct matrix* nm = emptyMatrD(n, n, permanency);
    for (int i=0; i<n; i++) {
        setElem(nm, i, i, 1);
    }
    return nm;
}



int addProdOfM(struct matrix* dest, struct matrix* src, double factor) {
    /*
    Function maps dest matrix -> dest + factor * src.
    */
    if ((dest->ncol!=src->ncol) || (dest->nrow != src->nrow)) {
        Cdelete(src);
        return 0;
    }
    for (int row = 0; row < dest->nrow; row++) {
        for (int col = 0; col < dest->ncol; col++) {
            dest->matrArray[row][col] += factor * src->matrArray[row][col];
        }
    }
    Cdelete(src);
    return 1;
}
int initMatrixTo(struct matrix* m, double elem) {
    /*
    Function initializes all elements of matrix to given argument
    */
    for (int row = 0; row < m->nrow; row++) {
        for (int col = 0; col < m->ncol; col++) {
            m->matrArray[row][col] = elem;
        }
    }
    return 1;
}

struct matrix* newMatrD(int rows, int cols, ...) {
    /* New matrix from multiple column vectors*/
    va_list args;
    va_start(args, cols);

    struct matrix *nm = emptyMatrD(rows, cols, 1);

    for (int col = 0; col < cols; col++) {
        struct vector* v = va_arg(args, struct vector*);
        if (!v) continue;
        for (int row = 0; row < rows; row++) {
            nm->matrArray[row][col] = v->vectorArray[row];
        }
        CdeleteV(v);
    }
    va_end(args);
    return nm;
}

int printMatrix(struct matrix* m1) {
    /*
    Function prints given matrix to terminal in nice form
    */
    printf("\n");
    for (int row=0; row<m1->nrow; row++) {
        for (int col=0; col<m1->ncol; col++) {
            printf("%3f ", m1->matrArray[row][col]);
        }
        printf("\n");
    }
    printf("\n");
    Cdelete(m1);
    return 1;
}

int memCpyM(struct matrix* dest, struct matrix* src) {
    /*
    Function copies src matrix to dest.
    */
    if ((dest->ncol!=src->ncol) || (dest->nrow != src->nrow)) {
        Cdelete(src);
        return 0;
    }
    for (int row = 0; row < dest->nrow; row++) {
        for (int col = 0; col < dest->ncol; col++) {
            dest->matrArray[row][col] = src->matrArray[row][col];
        }
    }
    Cdelete(src);
    return 1;
}

// Transposes:
struct matrix* transposeFiller(struct matrix *m1, struct matrix *tp) {
    /*Helper function for transpose operation*/
    for (int col = 0; col<m1->nrow; col++) {
        for (int row=0; row <m1->ncol; row++) {
            tp->matrArray[row][col] = m1->matrArray[col][row];
        }
    }
    return tp;
}
struct matrix* t(struct matrix* m1) {
    /*
    Function returns non-permanent transpose of original
    */
    struct matrix* tp = emptyMatrD(m1->ncol, m1->nrow, 0);
    transposeFiller(m1, tp);
    Cdelete(m1);
    return tp;
}
struct matrix* tP(struct matrix* m1) {
    /*
    Function returns permanent transpose of original matrix
    */
    struct matrix* tp = emptyMatrD(m1->ncol, m1->nrow, 1);
    transposeFiller(m1, tp);
    Cdelete(m1);
    return tp;
}


struct matrix* mulFiller(struct matrix* m1, struct matrix *m2, struct matrix *nm) {
    /*Helper function for multiplications*/
    float item = 0;
    for (int row = 0; row<nm->nrow; row++) {
        for (int col = 0; col<nm->ncol; col++) {
            item = 0;
            for (int iter=0; iter<m1->ncol; iter++) {
                item += m1->matrArray[row][iter] * m2->matrArray[iter][col];
            }
            nm->matrArray[row][col] = item;
        }
    }
    return nm;
}

struct matrix* mul(struct matrix *m1, struct matrix *m2) {
    /*
    Function returns m1*m2 as non-permanent matrix
    */
    if (m1->ncol==m2->nrow) {
        struct matrix* nm = emptyMatrD(m1->nrow, m2->ncol, 0);
        mulFiller(m1, m2, nm);
        Cdelete(m1);
        Cdelete(m2);
        return nm;
    }
    printf("Cannot multiply %dx%d with %dx%d matrix\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
    Cdelete(m1);
    Cdelete(m2);
    return NULL;
}

struct matrix* mulP(struct matrix *m1, struct matrix *m2) {
    /*
    Function returns m1*m2 as permanent matrix
    */
    if (m1->ncol==m2->nrow) {
        struct matrix* nm = emptyMatrD(m1->nrow, m2->ncol, 1);
        mulFiller(m1, m2, nm);
        Cdelete(m1);
        Cdelete(m2);
        return nm;
    }
    printf("Cannot multiply %dx%d with %dx%d matrix\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
    Cdelete(m1);
    Cdelete(m2);
    return NULL;
}

struct matrix* addFiller(struct matrix *m1, struct matrix *m2, struct matrix *nm) {
    /*Helper function for add operation*/
    for (int row=0; row<m1->nrow;row++) {
        for (int col=0; col<m1->ncol; col++) {
            nm->matrArray[row][col]=m1->matrArray[row][col]+m2->matrArray[row][col];
        }
    }
    return nm;
}
struct matrix* add(struct matrix* m1, struct matrix* m2) {
    /*
    Function returns m1 + m2 as non-permanent matrix
    */
    if ((m1->ncol==m2->ncol)&&(m1->nrow==m2->nrow)) {
        struct matrix* nm = emptyMatrD(m1->nrow, m1->ncol, 0);
        addFiller(m1, m2, nm);
        Cdelete(m1);
        Cdelete(m2);
        return nm;
    }
    printf("Cannot add %dx%d with %dx%d matrix\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
    Cdelete(m1);
    Cdelete(m2);
    return NULL;
}
struct matrix* addP(struct matrix* m1, struct matrix* m2) {
    /*
    Function returns m1 + m2 as permanent matrix
    */
    if ((m1->ncol==m2->ncol)&&(m1->nrow==m2->nrow)) {
        struct matrix* nm = emptyMatrD(m1->nrow, m1->ncol, 1);
        addFiller(m1, m2, nm);
        Cdelete(m1);
        Cdelete(m2);
        return nm;
    }
    printf("Cannot add %dx%d with %dx%d matrix\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
    Cdelete(m1);
    Cdelete(m2);
    return NULL;
}

struct matrix* dotMulFiller(struct matrix *m1, struct matrix *m2, struct matrix *nm) {
    /*
    Helper function for elementwise "naive" multiplication.
    */
    for (int row=0; row<m1->nrow;row++) {
        for (int col=0; col<m1->ncol; col++) {
            nm->matrArray[row][col]=m1->matrArray[row][col]*m2->matrArray[row][col];
        }
    }
    return nm;
}

struct matrix* dotMul(struct matrix* m1, struct matrix* m2) {
    /*
    Function returns non-permanent matrix from elementwise product
    of m1 and m2
    */
    if ((m1->ncol==m2->ncol)&&(m1->nrow==m2->nrow)) {
        struct matrix* nm = emptyMatrD(m1->nrow, m1->ncol, 0);
        dotMulFiller(m1, m2, nm);
        Cdelete(m1);
        Cdelete(m2);
        return nm;
    }
    printf("Cannot add %dx%d with %dx%d matrix\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
    Cdelete(m1);
    Cdelete(m2);
    return NULL;
}
struct matrix* dotMulP(struct matrix* m1, struct matrix* m2) {
    /*
    Function returns permanent matrix from elementwise product
    of m1 and m2
    */
    if ((m1->ncol==m2->ncol)&&(m1->nrow==m2->nrow)) {
        struct matrix* nm = emptyMatrD(m1->nrow, m1->ncol, 1);
        dotMulFiller(m1, m2, nm);
        Cdelete(m1);
        Cdelete(m2);
        return nm;
    }
    printf("Cannot add %dx%d with %dx%d matrix\n", m1->nrow, m1->ncol, m2->nrow, m2->ncol);
    Cdelete(m1);
    Cdelete(m2);
    return NULL;
}
long double SSELEM(struct matrix* m) {
    /*Function returns sum of squared elements of matrix*/
    long double s= 0;
    long double elemToSquare = 0;
    for (int row=0; row<m->nrow;row++) {
        for (int col=0; col<m->ncol; col++) {
            elemToSquare = (long double)m->matrArray[row][col];
            s += elemToSquare*elemToSquare;
        }
    }
    Cdelete(m);
    return s;
}

long double LOGSSELEM(struct matrix* m) {
    /*Function returns log of sum of squared elements in matrix*/
    long double s= 0;
    long double elemToSquare = 0;
    for (int row=0; row<m->nrow;row++) {
        for (int col=0; col<m->ncol; col++) {
            elemToSquare = (long double)m->matrArray[row][col];
            s += elemToSquare*elemToSquare;
        }
    }
    Cdelete(m);
    return logl(s);
}

struct matrix* addRetProdOfM(struct matrix* m1, struct matrix* m2, double factor) {
    /*
    Function returns non-permanent form of sum m1 + m2*factor
    */
    struct matrix* nm = emptyMatrD(m1->nrow, m1->ncol, 0);
    for (int row=0; row<nm->nrow;row++) {
        for (int col=0; col<nm->ncol; col++) {
            setElem(nm, row, col, m1->matrArray[row][col]+factor*m2->matrArray[row][col]);
        }
    }
    Cdelete(m1);
    Cdelete(m2);
    return nm;
}

struct matrix* rndMatrix(int lower, int upper, int rows, int cols, int permanency) {
    /*
    Function creates matrix with pseudo-random elements (from uniform distribution)
    within range [lower, upper]
    */
    struct matrix* nm = emptyMatrD(rows, cols, permanency);
    for (int row=0; row<rows;row++) {
        for (int col=0; col<cols; col++) {
            nm->matrArray[row][col] = (float)(((float)rand()/RAND_MAX)*(upper-lower)+lower);
        }
    }
    return nm;
}
struct matrix* createCopy(struct matrix* m) {
    /*
    Permanent copy of matrix m.
    */
    struct matrix* nm = emptyMatrD(m->nrow, m->ncol, 1);
    for (int row=0;row<m->nrow; row++) {
        for (int col=0;col<m->ncol; col++) {
            nm->matrArray[row][col] = m->matrArray[row][col];
        }
    }
    return nm;
}

struct matrix* addRow(struct matrix* m, int srcRow, int dstRow, float factor) {
    /*
    Function adds row 'srcRow' to 'destRow', factor times.
    */
    for (int coliter = 0; coliter<m->ncol; coliter++) {
        m->matrArray[dstRow][coliter] += factor * m->matrArray[srcRow][coliter];
    }
    return m;
}

struct matrix* addFLUdecomp(struct matrix* l, int row, int col, float factor) {
    /*
    Helper function for LU decomposition. Function is used to insert elements to lower triangle matrix
    */
    l->matrArray[row][col] = factor;
    return l;
}

struct matrix* mulRow(struct matrix* m, int Row, float factor) {
    /*
    Row 'Row' of matrix m is multiplied by factor 'factor'
    */
    for (int coliter = 0; coliter<m->ncol; coliter++) {
        m->matrArray[Row][coliter] *= factor;
    }
    return m;
}


struct matrix* linSolverS(struct matrix* m, struct matrix* t, int permanency) {
    /*
    Function solves Mx=t for x and returns it with given permanency.
    Solving is done by using Gaussian elimination.
    */
    if ((m->ncol != m->nrow)||((m->ncol!=t->nrow)||(m->ncol<t->ncol))) {
        Cdelete(m);
        Cdelete(t);
        return NULL;
    }
    struct matrix* sol = emptyMatrD(t->nrow, t->ncol, permanency);
    struct matrix* parSol;
    struct matrix* ma;
    float mul;
    for (int sCol = 0; sCol<sol->ncol; sCol++) {
        parSol = emptyMatrD(t->nrow, 1, 1);
        ma = createCopy(m);
        for (int iterRow = 0;iterRow<sol->nrow;iterRow++) {
            parSol->matrArray[iterRow][0] = t->matrArray[iterRow][sCol];
        }
        for (int iterCol = 0; iterCol<sol->ncol;iterCol++) {
            for (int iterRow = iterCol + 1; iterRow<sol->nrow; iterRow++) {
                mul = -elem(ma, iterRow, iterCol) / elem(ma, iterCol, iterCol);
                addRow(ma, iterCol, iterRow, mul);
                addRow(parSol, iterCol, iterRow, mul);
            }
        }
        for (int iterCol = ma->ncol-1; iterCol>0;iterCol--) {
            for (int iterRow = iterCol -1 ; iterRow >= 0; iterRow--) {
                mul = -elem(ma, iterRow, iterCol) / elem(ma, iterCol, iterCol);
                addRow(ma, iterCol, iterRow, mul);
                addRow(parSol, iterCol, iterRow, mul);
            }
        }
        for (int e = 0; e<sol->ncol; e++) {
            mul = 1/elem(ma, e, e);
            mulRow(ma, e, mul);
            mulRow(parSol, e, mul);
        }
        for (int row=0; row<sol->nrow; row++) {
            sol->matrArray[row][sCol]=parSol->matrArray[row][0];
        }
        deleteM(parSol);
        deleteM(ma);

    }

    Cdelete(m);
    Cdelete(t);
    return sol;
}

struct matrix* linSolverSLU(struct matrix* m, struct matrix* t, int permanency) {
    /*
    Function solves Mx=t for x and returns it with given permanency.
    Solving is done by using LU-decomposition
    */
    if ((m->ncol != m->nrow)||((m->ncol!=t->nrow)||(m->ncol<t->ncol))) {
        Cdelete(m);
        Cdelete(t);
        return NULL;
    }
    struct matrix* sol = emptyMatrD(t->nrow, t->ncol, permanency);
    struct matrix* parSol = emptyMatrD(t->nrow, t->ncol, permanency);
    struct matrix* ma, *l;
    float factor;
    ma = createCopy(m);
    l = emptyMatrD(m->nrow, m->ncol, 1);
    for (int iterRow = 0;iterRow<sol->nrow;iterRow++) {
        setElem(l, iterRow, iterRow, 1.0);
    }

    for (int iterCol = 0; iterCol<m->ncol;iterCol++) {
        for (int iterRow = iterCol + 1; iterRow<sol->nrow; iterRow++) {
            factor = -elem(ma, iterRow, iterCol) / elem(ma, iterCol, iterCol);
            addRow(ma, iterCol, iterRow, factor);
            addFLUdecomp(l, iterRow, iterCol, -factor);
        }
    }
    float sum = 0;
    for (int iterRow = 0; iterRow < m->nrow; iterRow++) {
        for (int pSolCol=0; pSolCol<parSol->ncol;pSolCol++) {
            sum = 0;
            for (int iterCol=0; iterCol<iterRow; iterCol++) {
                sum += l->matrArray[iterRow][iterCol]*parSol->matrArray[iterCol][pSolCol];
            }
            setElem(parSol, iterRow, pSolCol, t->matrArray[iterRow][pSolCol]-sum);
        }
    }
    for (int iterRow = ma->nrow-1; iterRow >=0 ; iterRow--) {
        for (int solCol=0; solCol<sol->ncol;solCol++) {
            sum = 0;
            for (int iterCol = iterRow + 1; iterCol < ma->ncol; iterCol++) {
                sum += ma->matrArray[iterRow][iterCol]*sol->matrArray[iterCol][solCol];
            }
            setElem(sol, iterRow, solCol, (parSol->matrArray[iterRow][solCol] - sum) / ma->matrArray[iterRow][iterRow]);
        }
    }
    deleteM(parSol);
    deleteM(ma);
    deleteM(l);

    Cdelete(m);
    Cdelete(t);
    
    return sol;
}
