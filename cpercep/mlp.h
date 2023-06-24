#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "perceptron.h"

typedef struct multilayer_perceptron_t {
    
    // Training epochs:
    int epoch_count;

    // MLP Input:
    int input_count;
     
    // Input layer:
    int p_input_count;
    perceptron_t **p_input;
    double *p_input_output;
    
    // Hidden layer:
    int p_hidden1_count;
    perceptron_t **p_hidden1;
    double *p_hidden1_output;

    // Output layer:
    int p_output_count;
    perceptron_t **p_output;
    double *p_output_output;
    
} multilayer_perceptron_t;

multilayer_perceptron_t *init_mlp(int p_input_count, int p_hidden1_count, int p_output_count, int epoch_count);
void destroy_mlp(multilayer_perceptron_t *mlp);

void train_mlp(multilayer_perceptron_t *mlp, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate);

#endif