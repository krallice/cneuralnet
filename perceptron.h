#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Struct init and destruction:
typedef struct perceptron_t {
    int training_epoch_count;
    int input_count;
    double *weights;
    double bias_weight;
    double (*activation_function)(double);
    double (*derivative_activation_function)(double);
} perceptron_t;
perceptron_t *init_perceptron(const int input_count, double (*activation_function)(double), double (*derivative_activation_function)(double), int training_epoch_count);
void destroy_perceptron(perceptron_t *p);

// Activation functions:
double sign_activation_function(double x);
double step_activation_function(double x);

double linear_activation(double x);
double derivative_linear_activation(double x);

double relu_activation(double x);
double derivative_relu_activation(double x);

double leaky_relu_activation(double x);
double derivative_leaky_relu_activation(double x);

double sigmoid_activation(double x);
double derivative_sigmoid_activation(double x);

// For use in single node networks (singleton perceptron):
// Activate the perceptron, and return the result:
double perceptron_feedforward(perceptron_t *p, const double training_features[]);
// Used for single node networks:
void train_perceptron(perceptron_t *p, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate);

#endif