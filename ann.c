#include "matrixoperations.h"
#include "ann.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

int initLayer(struct ANNlayer* layer, int numOfNodes, struct ANNlayerFunction *prevfn,  struct ANNlayerFunction *nextfn) {
    /*
    Function that fills in the layer according to arguments.
    */
    layer->values = emptyMatrD(numOfNodes, 1, 1);
    initMatrixTo(layer->values, 0);
    layer->nF = nextfn;
    layer->pF = prevfn;
    layer->nOfNodes = numOfNodes;
    return 1;
}
int initANNFunction(struct ANNlayerFunction *lf, int prevNodes, int nextNodes) {
    /*
    Function that fills the functionlayers between normal layers.
    */

    // All coefficients in weight matrices are initialized to small numbers.
    lf->coefficients = rndMatrix(-1, 1, prevNodes, nextNodes, 1);
    lf->bias = rndMatrix(-1, 1, nextNodes, 1, 1);
    lf->pCoefGrad = rndMatrix(-1, 1, prevNodes, nextNodes, 1);
    lf->pBiasGrad = rndMatrix(-1, 1, nextNodes, 1, 1);
    lf->CGradSum = rndMatrix(0, 0, prevNodes, nextNodes, 1);
    lf->BGradSum = rndMatrix(0, 0, nextNodes, 1, 1);

    return 1;
}
struct ANNmodel* newANN(int numOfLayers, ...) {
    /*
    Function creates and returns new ANN model according to user inputs.
    First input is num of layers and rest are sizes for each layer.

    Size of first layer must match desired input size and similarly to last layer.
    */
    if (numOfLayers<2) {
        return NULL;
    }
    va_list nodes;
    va_start(nodes, numOfLayers);

    struct ANNmodel* newModel = malloc(sizeof(struct ANNmodel));
    struct ANNlayer* iterlayer = NULL;
    struct ANNlayer* nextlayer = malloc(sizeof(struct ANNlayer));
    newModel->start = nextlayer;
    struct ANNlayer* prevlayer = NULL;
    int prevLS;
    int nextLS = va_arg(nodes, int);
    newModel->totalNodes = 0;

    struct ANNlayerFunction* nextfunction;
    struct ANNlayerFunction* prevfunction = NULL;

    for (int layer = 0; layer < numOfLayers - 1; layer ++) {
        prevLS = nextLS;
        nextLS = va_arg(nodes, int);
        newModel->totalNodes += 2 * prevLS*nextLS;
        prevlayer = iterlayer;
        iterlayer = nextlayer;
        nextlayer = malloc(sizeof(struct ANNlayer));
        nextfunction = malloc(sizeof(struct ANNlayerFunction));
        initANNFunction(nextfunction, prevLS, nextLS);
        initLayer(iterlayer, prevLS, prevfunction, nextfunction);
        iterlayer->prev = prevlayer;
        iterlayer->next = nextlayer;
        // Initialize layer and function and add it to model.
        prevfunction = nextfunction; //For next layer, newly created function is previous function.
    }
    prevlayer = iterlayer;
    iterlayer = nextlayer;
    nextlayer = NULL;
    initLayer(iterlayer, nextLS, prevfunction, NULL);
    iterlayer->next = NULL;
    iterlayer->prev = prevlayer;
    newModel->end = iterlayer;
    newModel->nOfGCombined = 1;
    va_end(nodes);
    return newModel;
}


int deleteANN(struct ANNmodel* model) {
    /*
    Function deletes ANN model from dynamic memory
    */
    struct ANNlayer* iterlayer;
    struct ANNlayer* delLayer;
    iterlayer = model->start;
    if (iterlayer==NULL) {return 0;}
    do {
        if (iterlayer->nF!=NULL) {
            deleteM(iterlayer->nF->coefficients);
            deleteM(iterlayer->nF->bias);
            deleteM(iterlayer->nF->pCoefGrad);
            deleteM(iterlayer->nF->pBiasGrad);
            deleteM(iterlayer->nF->BGradSum);
            deleteM(iterlayer->nF->CGradSum);
            free(iterlayer->nF);
        }
        deleteM(iterlayer->values);
        delLayer = iterlayer;
        iterlayer = iterlayer->next;
        free(delLayer);
    }
    while (iterlayer != NULL);
    free(model);
    return 1;
}

struct ANNlayer* FP(struct matrix *input, struct ANNmodel* model) {
    /*
    Forward propagation for model and some input matrix.
    Function inputs argument matrix "input" to model and returns the output layer that contains
    information for output value.    

    Layer is returned instead of the output matrix so that backpropagation can be implemented by
    using the layer as argument for BP-function.
    */
    struct ANNlayer* ilayer = model->start;
    if (ilayer->nOfNodes != input->nrow) {
        printf("Invalid input size\n");
        Cdelete(input);
        return NULL;
    }
    memCpyM(ilayer->values, input);

    do {
        ilayer = ilayer->next;
        initMatrixTo(ilayer->values, 0);
        for (int newind = 0; newind<ilayer->nOfNodes; newind++) {
            for (int prevind = 0; prevind < ilayer->prev->nOfNodes; prevind++) {
                ilayer->values->matrArray[newind][0] += ilayer->pF->coefficients->matrArray[prevind][newind] * ilayer->prev->values->matrArray[prevind][0];       
            }
            ilayer->values->matrArray[newind][0] += ilayer->pF->bias->matrArray[newind][0];
        }
    }
    while (ilayer->next != NULL);
    Cdelete(input);
    return ilayer;
}

int emptySumGradients(struct ANNlayer * last) {
    struct ANNlayer *iter = last;
    while (iter->prev!=NULL) {
        initMatrixTo(iter->pF->BGradSum, 0);
        initMatrixTo(iter->pF->CGradSum, 0);
        iter = iter->prev;
    }    
    return 1;
}


int BP_MSE2_DONOTCALL(struct ANNlayer* lastLayer, struct matrix* target, struct ANNmodel* annmodel) {
    /* 
    Loss: MSE
    Funktio päivittää mallin gradienttia LISÄÄMÄLLÄ siihen gradienttimuutokset
    Tarkoitus on, että monta lisäämällä ja jakamalla stepin määrällä voi liikkua keskiarvogradientin suuntaan
    */ 
    struct ANNlayer *iterlayer = lastLayer; // Layeri, joka on ifunc edessä -> ifunc luo layerin nodeille arvot
    struct ANNlayerFunction* ifunc = iterlayer->pF; // Function joka luo seuraavaan aktivaatioarvot
    double addhelper;

    //Lisätään viimeiseen layeriin -1* vertailuvektori -> Saadaan epsilonvektori
    addProdOfM(iterlayer->values, target, -1.0); 

    for (int nextNode = 0; nextNode < iterlayer->nOfNodes; nextNode++) {
        // Iteroidaan jokainen epsilon läpi.

        // Lisätään jokaiseen biasgradienttiin indeksissä i, 2*epsilon_i, sekä samat myös sum of total differencesiin.
        ifunc->pBiasGrad->matrArray[nextNode][0] = 2 * iterlayer->values->matrArray[nextNode][0];
        annmodel->sumOfDifferences += 2*iterlayer->values->matrArray[nextNode][0];
    }

    for (int prevNode=0;prevNode<iterlayer->prev->nOfNodes; prevNode++) {
        for (int nextNode=0;nextNode<iterlayer->nOfNodes; nextNode++) {
            addhelper = iterlayer->prev->values->matrArray[prevNode][0]*ifunc->pBiasGrad->matrArray[nextNode][0];
            // Coefgradienttin a^L=v(L)*biasgradientti
            ifunc->pCoefGrad->matrArray[prevNode][nextNode] = addhelper;
            annmodel->sumOfDifferences += addhelper;
        }    
    }
    // Poistetaan sivuvaikutus
    addProdOfM(iterlayer->values, target, 1.0);

    // Iterointi voindaan aloittaa.
    iterlayer = iterlayer->prev;
    while (iterlayer->prev != NULL) {
        ifunc = iterlayer->pF;

        for (int prevNode = 0; prevNode < iterlayer->prev->nOfNodes; prevNode++) {
            // iteroidaan funtiosta edellinen layer
            for (int nextNode = 0; nextNode < iterlayer->nOfNodes; nextNode++) {
                //Iteroidaan seuraavaa layer funktiosta
                addhelper = 0;
                for (int nextnextNode=0; nextnextNode<iterlayer->next->nOfNodes; nextnextNode++) {
                    // Seuraavan layerin z funktion derivaattaan vaikuttaa sitäseuraavan arvot
                    addhelper += iterlayer->nF->coefficients->matrArray[nextNode][nextnextNode];
                }
                annmodel->sumOfDifferences += addhelper;
                ifunc->pBiasGrad->matrArray[nextNode][0] = addhelper;
                addhelper *= iterlayer->prev->values->matrArray[prevNode][0];
                annmodel->sumOfDifferences += addhelper;
                ifunc->pCoefGrad->matrArray[prevNode][nextNode] = addhelper;
            }
        }
        iterlayer = iterlayer->prev;
    }
    iterlayer = lastLayer;
    while (iterlayer->prev != NULL) {
        ifunc = iterlayer->pF;
        addProdOfM(ifunc->BGradSum, ifunc->pBiasGrad, 1);
        addProdOfM(ifunc->CGradSum, ifunc->pCoefGrad, 1);
        iterlayer = iterlayer->prev;
    }
    return 0;
}


int BP_MSE(struct ANNlayer* lastLayer, struct matrix* target, struct ANNmodel* annmodel) {
    /*
    BackPropagation for singular input, output pair.
    */
    annmodel->sumOfDifferences = 0;
    annmodel->nOfGCombined = 1;
    emptySumGradients(lastLayer);
    return BP_MSE2_DONOTCALL(lastLayer, target, annmodel);
}

int gradientDescent(struct ANNmodel* model, double lRate) {
    /*
    Muuttaa mallin parametreja gradientista poispäin lRate verran.
    */
    struct ANNlayer* iterlayer = model->start;
    while (iterlayer->nF != NULL) {
        addProdOfM(iterlayer->nF->coefficients, iterlayer->nF->CGradSum, -lRate);
        addProdOfM(iterlayer->nF->bias, iterlayer->nF->BGradSum, -lRate);
        iterlayer = iterlayer->next;
    }
    return 0;
}

int updateANNGradientFromData(struct pDataFormatANN *data, struct ANNmodel* m) {
    /*
    Function calculates the mean gradient for model parameters respect to MSE loss function.
    */
    m->sumOfDifferences = 0;
    m->nOfGCombined = data->nOfData;
    emptySumGradients(m->end);
    struct matrix* input;
    struct matrix* target;
    for (int iter = 0; iter < data->nOfData; iter++) {
        // For each input, output pair in given datastructure, update the sumgradient in the model
        input = data->inputs[iter];
        target = data->outputs[iter];
        BP_MSE2_DONOTCALL(FP(input, m), target, m);
    }
    m->sumOfDifferences /= m->nOfGCombined;
    return 1;
}

long double rErrorSum(struct pDataFormatANN *data, struct ANNmodel* m) {
    /*
        Function calculates the average sum of squared errors for (input, output) pairs in data
    */
    struct matrix* input;
    struct matrix* target;
    long double cSum=0.0;
    for (int iter=0; iter<data->nOfData; iter++) {
        input = data->inputs[iter];
        target = data->outputs[iter];
        cSum += SSELEM(addRetProdOfM(FP(input, m)->values, target, -1));
    }
    return cSum / (long double)data->nOfData;
}

int trainModelUntilConverge(struct ANNmodel* m, struct pDataFormatANN* data) {
    /*
        Function 
        1. Updates gradient calculations for model m,
        2. Updates model parameters with gradient descent method.
        Step size is adjusted automatically.

        The steps above are repeated until converge criteria are met or
        the maxiterations are reached.
    */
    long double maxerror = rErrorSum(data, m);
    long double itererror = maxerror;
    double lr = 0.01;
    double minlr = (double)(1.0E-20);
    int maxiterations = 10000;
    double step;
    double gl;
    int state;
    for (int i = 0; i < maxiterations; i++) {
        printf("#Epoch: %d\n", i);
        updateANNGradientFromData(data, m);
        itererror = maxerror;
        state=0;
        while ((lr>=minlr)&&(state==0)){
            gl = m->sumOfDifferences;
            if (gl<0) {gl = -gl;} /* Gl is the abs value of the sum of differences*/
            if (gl==0) {
                printf("Converged...\n");
                break;
            }
            step = lr/(gl); /*Stepsize is adjusted accoding to learn rate and affect of change (sum of differences)*/

            gradientDescent(m, step);
            itererror = rErrorSum(data, m);
            printf("Gradient size: %f, step: %f, ErrorDiff: %.20Lf, lr: %f\n", gl, step, maxerror - itererror, lr);
            if (itererror<=maxerror) {
                maxerror = itererror;
                lr *= sqrt(2);
                state=1;
            } else {
                gradientDescent(m, -step);
                lr = lr / 3.14159;
            }
        }
        if (lr<minlr) {
            printf("Iteration stopped due to hitting minlr\n");
            break;
        }
        if (gl==0) {
            break;
        }
    }
    
    return 1;
}

void loadDataset(const char* filename, struct pDataFormatANN* dataset, int numSamples, int inputLen, int outputLen) {
    /*
    Function opens textfile and imports data if form "<ip1> <ip2> ... <ipn> | <op1> ... <opm>\n"
    where ip1, ip2, ... are individual elemets of input vector and op_i's are elements of output vector
    Data structure pDataFormatANN is filled with data.

    Purpose of function is to be able to import training and test datasets during development and testing.
    */
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Tiedoston avaaminen epäonnistui");
        return;
    }
    dataset->nOfData = numSamples;
    dataset->inputLen = inputLen;
    dataset->outputLen = outputLen;
    dataset->inputs = (struct matrix**)malloc(numSamples * sizeof(struct matrix*));
    dataset->outputs = (struct matrix**)malloc(numSamples * sizeof(struct matrix*));

    for (int i = 0; i < numSamples; i++) {
        dataset->inputs[i] = emptyMatrD(inputLen, 1, 1);
        dataset->outputs[i] = emptyMatrD(outputLen, 1, 1);
        
        for (int j = 0; j < inputLen; j++) {
            fscanf(file, "%f ", &dataset->inputs[i]->matrArray[j][0]);
        }
        fgetc(file); // Skip '|' character
        for (int j = 0; j < outputLen; j++) {
            fscanf(file, " %f", &dataset->outputs[i]->matrArray[j][0]);
        }
        fgetc(file); // '\n' character.
    }
    fclose(file);
}
int deleteDataset(struct pDataFormatANN* data) {
    /*Function deletes the created dataset from dynamic memory*/
    for (int iter=0; iter<data->nOfData; iter++) {
        deleteM(data->inputs[iter]);
        deleteM(data->outputs[iter]);
    }
    free(data->inputs);
    free(data->outputs);
    free(data);
    return 1;
}

int main(void) {
    /* New model with 2 layers; sized 4 and 2*/
    struct ANNmodel* m = newANN(2, 4, 2);
    /*Empty datastructure for training and testing data*/
    struct pDataFormatANN *data = malloc(sizeof(struct pDataFormatANN));

    /*Num of rows to be imported from file*/
    int n = 100;
    /*Load n rows from file training_data.txt, input size 4, output size 2*/
    loadDataset("training_data.txt", data, n, 4, 2);

    /*Train model with data*/
    trainModelUntilConverge(m, data);
    
    /*Compare model outputs to real "target" ouput set by printing some results*/
    struct matrix* input;
    struct matrix* output;
    for (int i=0; i<((n>3) ? 3 : n); i++) {
        input = data->inputs[i];
        output = data->outputs[i];
        printf("\n\nTestresult #%d\n", i+1);
        printf("Estimate:");
        printMatrix(FP(input, m)->values);
        printf("Target output:");
        printMatrix(output);
    }

    deleteANN(m);
    deleteDataset(data);
    return 1;

}

/*
Compiling instructions:

GCC: gcc -g -Wall -o  project ann.c matrixoperations.c -lm
RUN: ./project

Valgrind: valgrind --leak-check=full  ./project
*/