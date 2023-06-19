#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "perceptron.h"

void model_x_gt_9(void) {

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

    // // Predict and check, non-integer shows the limitation
    // // of a single perceptron trained on integers
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
    return;
}

int main(void) {

    // Random Seed:
    srand(time(NULL)); 

    // A single input perceptron, trained on the dataset of x > 9:
    model_x_gt_9();

    return 0;
}