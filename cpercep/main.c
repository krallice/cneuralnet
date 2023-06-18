#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

typedef struct perceptron_t {
    int training_epoch_count;
    int input_count;
    double *weights;
    double bias_weight;
    double (*activation_function)(double);
} perceptron_t;

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

void destroy_perceptron(perceptron_t *p) {
    free(p->weights);
    free(p);
    return;
}

int main(void) {

    // Random Seed:
    srand(time(NULL)); 

    // Modelling x > 9:
    const double training_features[][1] = {
        {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9},
        {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}
    };
    const double training_labels[] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1
        };
    int training_rows = sizeof(training_features) / sizeof(training_features[0]);
    int training_columns = sizeof(training_features[0]) / sizeof(training_features[0][0]);

    perceptron_t *p = init_perceptron(1, sign_activation_function, 1000);

    train_perceptron(p, training_rows, training_columns, training_features, training_labels, 0.1);

    // Predict and check:
    for (int i = -30; i < 31; i++) {
        const double prediction_features[] = {i};
        double prediction = perceptron_feedforward(p, prediction_features);
        double expected_value = i > 9 ? 1 : -1;
        printf("STATUS: %s Feature: %d Prediction: %f\n", expected_value == prediction ? "SUCCESS" : "FAILURE", i, prediction);
    }

    // // Predict and check:
    // for (double i = 8; i < 10; i += 0.1) {
    //     const double prediction_features[] = {i};
    //     double prediction = perceptron_feedforward(p, prediction_features);
    //     double expected_value = i > 9 ? 1 : -1;
    //     printf("STATUS: %s Feature: %f Prediction: %f\n", expected_value == prediction ? "SUCCESS" : "FAILURE", i, prediction);
    // }

    // Dump state:
    printf("Perceptron Bias: %f\n", p->bias_weight);
    printf("Perceptron Weight: %f\n", p->weights[0]);

    destroy_perceptron(p);
    return 0;
}