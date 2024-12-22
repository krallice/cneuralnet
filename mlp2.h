#ifndef MLP2_H
#define MLP2_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "perceptron.h"

typedef struct multilayer_perceptron2_t {
    
    // Training epochs:
    int epoch_count;

    // MLP Input:
    int input_count;
    
    // Hidden layer 1:
    int p_hidden1_count;
    perceptron_t **p_hidden1;
    double *p_hidden1_output;

    // Hidden layer 2:
    int p_hidden2_count;
    perceptron_t **p_hidden2;
    double *p_hidden2_output;

    // Output layer:
    int p_output_count;
    perceptron_t **p_output;
    double *p_output_output;
    
} multilayer_perceptron2_t;

multilayer_perceptron2_t *init_mlp2(int p_input_count, int p_hidden1_count, int p_hidden2_count, int p_output_count, 
    double (*hidden1_activation_function)(double), double (*hidden1_derivative_activation_function)(double),
    double (*hidden2_activation_function)(double), double (*hidden2_derivative_activation_function)(double),
    double (*output_activation_function)(double),  double (*output_derivative_activation_function)(double), int epoch_count);
void destroy_mlp2(multilayer_perceptron2_t *mlp2);

void mlp2_feedforward(multilayer_perceptron2_t *mlp, const double training_features[]);
void mlp2_backpropagate(multilayer_perceptron2_t *mlp, const double training_features[], const double training_labels[], const double learning_rate);

void train_mlp2(multilayer_perceptron2_t *mlp, int feature_count, int feature_dimension, const double training_features[feature_count][feature_dimension],
    int label_dimension, const double training_labels[feature_count][label_dimension], const double learning_rate);

void save_mlp2_weights(const multilayer_perceptron2_t *mlp, const char *filename);
void load_mlp2_weights(multilayer_perceptron2_t *mlp, const char *filename);

#endif
