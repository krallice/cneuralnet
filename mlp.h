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
    perceptron_t **p_hidden1;
    double *p_hidden1_output;

    // Output layer:
    int p_output_count;
    perceptron_t **p_output;
    double *p_output_output;
    
} multilayer_perceptron_t;

multilayer_perceptron_t *init_mlp(int p_input_count, int p_hidden1_count, int p_output_count, 
    double (*hidden1_activation_function)(double), double (*hidden1_derivative_activation_function)(double), 
    double (*output_activation_function)(double),  double (*output_derivative_activation_function)(double), int epoch_count);
void destroy_mlp(multilayer_perceptron_t *mlp);

void mlp_feedforward(multilayer_perceptron_t *mlp, const double training_features[]);
void mlp_backpropagate(multilayer_perceptron_t *mlp, const double training_features[], const double training_labels[], const double learning_rate);

void train_mlp(multilayer_perceptron_t *mlp, int feature_count, int feature_dimension, const double training_features[feature_count][feature_dimension],
    int label_dimension, const double training_labels[feature_count][label_dimension], const double learning_rate);

void save_mlp_weights(const multilayer_perceptron_t *mlp, const char *filename);
void load_mlp_weights(multilayer_perceptron_t *mlp, const char *filename);

#endif
