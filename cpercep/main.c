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

void model_linear(void) {
    
    const int feature_count = 20;
    const int point_lower_bound = -20;
    const int point_upper_bound = 20;

    double training_features[feature_count][2];
    double training_labels[feature_count];

    // Generate the test data:
    for (int i = 0; i < feature_count; i++) {

        // x co-ord:
        training_features[i][0] = point_lower_bound + rand() % (point_upper_bound - point_lower_bound + 1);
        // y co-ord:
        training_features[i][1] = point_lower_bound + rand() % (point_upper_bound - point_lower_bound + 1);

        // Model linear seperation using
        // y = x/2 + 5, if we are above the line, fire
        int y = (training_features[i][0] / 2) + 5;
        training_labels[i] = (training_features[i][1] >= y) ? 1 : -1;
    
        printf("Generated Random Point (%d/%d): (x,y): (%d, %d). Point is %s the line.\n", i + 1, feature_count, 
            (int)training_features[i][0], (int)training_features[i][1],
            training_labels[i] == 1 ? "above" : "below");
    }

    perceptron_t *p = init_perceptron(2, sign_activation_function, 100000);

    train_perceptron(p, feature_count, 2, training_features, training_labels, 0.1);

    printf("=== End Training ===\n");

    const int predict_count = 20;
    for (int i = 0; i < predict_count; i++) {

        double predict_features[2];
        int predict_feature_label;

        // x co-ord:
        predict_features[0] = point_lower_bound + rand() % (point_upper_bound - point_lower_bound + 1);
        // y co-ord:
        predict_features[1] = point_lower_bound + rand() % (point_upper_bound - point_lower_bound + 1);

        // Generate result:
        int y = (predict_features[0] / 2) + 5;
        predict_feature_label = (predict_features[1] >= y) ? 1 : -1;

        double prediction = perceptron_feedforward(p, predict_features);

        printf("Point (%d,%d) is %s the line. Prediction was: %s\n", 
        (int)predict_features[0], (int)predict_features[1], (predict_feature_label == 1) ? "above" : "below",
        (predict_feature_label == prediction) ? "CORRECT" : "xx Incorrect xx");

    }

    destroy_perceptron(p);
    return;
}

void model_AND(void) {

    // Modelling logical AND:
    const double training_features[][2] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const double training_labels[] = {
        -1, -1, -1, 1
    };
    int training_rows = sizeof(training_features) / sizeof(training_features[0]);
    int training_columns = sizeof(training_features[0]) / sizeof(training_features[0][0]);

    perceptron_t *p = init_perceptron(training_columns, sign_activation_function, 100);

    train_perceptron(p, training_rows, training_columns, training_features, training_labels, 0.1);

    for (int i = 0; i < training_rows; i++) {
        double prediction = perceptron_feedforward(p, training_features[i]);
        printf("ValueA: %0.1f ValueB: %0.1f Expected: %0.1f Prediction: %0.1f\n", training_features[i][0], training_features[i][1], 
            (double)((int)training_features[i][0] & (int)training_features[i][1]), (prediction + 1) * 0.5);
    }

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
    // model_x_gt_9();

    // A dual input perceptron, trained to operate as an AND gate:
    //model_AND();

    // A dual input perceptron, trained on a linear equation:
    model_linear();

    return 0;
}