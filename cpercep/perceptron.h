#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct perceptron_t {
    int training_epoch_count;
    int input_count;
    double *weights;
    double bias_weight;
    double (*activation_function)(double);
} perceptron_t;

perceptron_t *init_perceptron(const int input_count, double (*activation_function)(double), int training_epoch_count);
void destroy_perceptron(perceptron_t *p);

double sign_activation_function(double x);
double sigmoid_activation(double x);

double perceptron_feedforward(perceptron_t *p, const double training_features[]);

void train_perceptron(perceptron_t *p, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate);

#endif