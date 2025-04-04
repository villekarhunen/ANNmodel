#ifndef MATRIXOPERATIONS_H
#define MATRIXOPERATIONS_H



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

// Function Prototypes
struct matrix* newMatrD(int rows, int cols, ...);
struct matrix* I(int n, int permanency);
struct matrix* emptyMatrD(int rows, int cols, int permanency);
struct vector* newVecD(int size, ...);
int deleteV(struct vector* v);
int deleteM(struct matrix* m);
int Cdelete(struct matrix* m);
long double SSELEM(struct matrix* m);
long double LOGSSELEM(struct matrix* m);

int memCpyM(struct matrix* dest, struct matrix* src);
int addProdOfM(struct matrix* dest, struct matrix* src, double factor);
struct matrix* addRetProdOfM(struct matrix* dest, struct matrix* src, double factor);
int initMatrixTo(struct matrix* m, double elem);

int Cdelete(struct matrix* m);
int CdeleteV(struct vector* v);
int printMatrix(struct matrix* m);
struct matrix* t(struct matrix* m);
struct matrix* tP(struct matrix* m);
struct matrix* mul(struct matrix* m1, struct matrix* m2);
struct matrix* mulP(struct matrix* m1, struct matrix* m2);
struct matrix* add(struct matrix* m1, struct matrix* m2);
struct matrix* addP(struct matrix* m1, struct matrix* m2);
struct matrix* dotMul(struct matrix* m1, struct matrix* m2);
struct matrix* dotMulP(struct matrix* m1, struct matrix* m2);
struct matrix* rndMatrix(int lower, int upper, int rows, int cols, int permanency);
struct matrix* setElem(struct matrix *m, int row, int col, float newElem);
struct matrix* linSolverS(struct matrix* m, struct matrix* t, int permanency);
struct matrix* linSolverSLU(struct matrix* m, struct matrix* t, int permanency);


#endif  // MATRIXOPERATIONS_H
