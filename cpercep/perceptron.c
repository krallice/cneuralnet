#include "perceptron.h"

perceptron_t *init_perceptron(const int input_count, double (*activation_function)(double), int training_epoch_count) {
    
    // Init and zeroise:
    perceptron_t *p = (perceptron_t*)malloc(sizeof(*p));
    memset(p, 0, sizeof(*p));

    // Allocate the weights array
    // Randomly set the starting weights on a value between -1 and 1
    p->weights = malloc(sizeof(double) * input_count);
    for (int i = 0; i < input_count; i++) {
        p->weights[i] = (rand() / (double)RAND_MAX * 2 - 1);
    }

    // Assign the Biases weight between -1 and 1:
    p->bias_weight = (rand() / (double)RAND_MAX * 2 - 1);

    p->input_count = input_count;
    p->activation_function = activation_function;
    p->training_epoch_count = training_epoch_count;

    return p;
}

void destroy_perceptron(perceptron_t *p) {
    free(p->weights);
    free(p);
    return;
}

double sign_activation_function(double x) {
    return x < 0 ? -1 : 1;
}

double perceptron_feedforward(perceptron_t *p, const double training_features[]) {
    // Weighted sum up all training features, and run it through our activation function.
    // As the bias input is always 1, we can just start off straight with the bias value:
    double weighted_sum = p->bias_weight;
    for (int i = 0; i < p->input_count; i++) {
        weighted_sum += training_features[i] * p->weights[i];
    }
    return p->activation_function(weighted_sum);
}

void train_perceptron(perceptron_t *p, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate) {

    // Exit if trying to train based on more features than expected:
    if (p->input_count != column_count) {
        printf("Invalid Input\n");
        return;
    }

    for (int epoch = 0; epoch < p->training_epoch_count; epoch++) {
        // Foreach dataset:
        for (int i = 0; i < row_count; i++) {
            
            // Create a prediction for that dataset:
            double prediction = perceptron_feedforward(p, training_features[i]);

            // Adjust the weights, aka train the model:

            // First of all, calculate the error. That is to say,
            // how far off was the predicted label from the correct label?
            // Predicted, Correct, Error
            //  1    1    0 (correct response)
            //  1   -1    -2 (incorrect, was a false positive)
            // -1    1    2 (incorrect. was a false negative)
            // -1   -1    0 (correct response)
            double error_difference = training_labels[i] - prediction;

            // Nudge the bias weight in the direction of the error difference.
            // Use the learning_rate as a magic scalar that scales how drastically the weight is nudged. Lower values mean slower responses to training (but more stable):
            p->bias_weight += learning_rate * error_difference;

            // Nudge the rest of the input weights in the same way:
            for (int j = 0; j < p->input_count; j++) {

                p->weights[j] += learning_rate * error_difference * training_features[i][j];
            }
        }
    }

    return;
}