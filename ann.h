#ifndef ANNOPER_H
#define ANNOPER_H

#include "matrixoperations.h"
//#include <stdarg.h>

struct ANNlayerFunction {
    struct matrix* coefficients; // Some form of matrix consisting parameters for particular function layer.
    struct matrix* bias;
    struct matrix* CGradSum;
    struct matrix* BGradSum;
    struct matrix* pCoefGrad;
    struct matrix* pBiasGrad; // Partial gradient, used for updating functionality. Felt clear, might delete later. Same dims as in functionality
    // In case we want to add some momentum methods for converge, it must be added here or we must create new struct for that.
};

struct ANNlayer {
    int nOfNodes;
    struct ANNlayer* next; // Address of next layer. If NULL; then output
    struct ANNlayer* prev; //Address to prev layer. If NULL; then inputlayer
    struct ANNlayerFunction *nF; //Address to functionality between this and next layer;
    struct ANNlayerFunction *pF; //Address to functionality of between prev and this layer;
    struct matrix* values; // nOfNodes x 1 matrix consisting of node values, used for calculating forward propagation
};

struct ANNmodel {
    struct ANNlayer * start;
    double sumOfDifferences;
    int totalNodes;
    int nOfGCombined;
    struct ANNlayer * end;
};

struct pDataFormatANN {
    struct matrix ** inputs;
    struct matrix ** outputs;
    int nOfData;
    int inputLen;
    int outputLen;
};

struct ANNmodel* newANN(int numOfLayers, ...);
int deleteANN(struct ANNmodel* model);
struct ANNlayer* FP(struct matrix *input, struct ANNmodel* model);
int trainModelUntilConverge(struct ANNmodel* m, struct pDataFormatANN* data);
void loadDataset(const char* filename, struct pDataFormatANN* dataset, int numSamples, int inputLen, int outputLen);
int deleteDataset(struct pDataFormatANN* data);

#endif 