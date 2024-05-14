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
    
    // Hidden layer:
    int p_hidden1_count;
    double** p_hidden1_weights;
    double p_hidden1_bias_weight;

    // Hidden layer:
    int p_output_count;
    double* p_output_weights;
    double p_output_bias_weight;

    double (*hidden1_activation_function)(double);
    double (*hidden1_derivative_activation_function)(double);
    double (*output_activation_function)(double);
    double (*output_derivative_activation_function)(double);
    
} multilayer_perceptron_t;

double step_function(double x);

multilayer_perceptron_t *init_mlp(int p_input_count, int p_hidden1_count, int p_output_count, 
    double (*hidden1_activation_function)(double), double (*hidden1_derivative_activation_function)(double), 
    double (*output_activation_function)(double),  double (*output_derivative_activation_function)(double), int epoch_count);
void destroy_mlp(multilayer_perceptron_t *mlp);

void train_mlp(multilayer_perceptron_t *mlp, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate);

#endif