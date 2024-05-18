#include "mlp.h"

multilayer_perceptron_t *init_mlp(int p_input_count, int p_hidden1_count, int p_output_count, 
    double (*hidden1_activation_function)(double), double (*hidden1_derivative_activation_function)(double), 
    double (*output_activation_function)(double),  double (*output_derivative_activation_function)(double), int epoch_count) {

    // Init and zeroise:
    multilayer_perceptron_t *mlp = (multilayer_perceptron_t*)malloc(sizeof(multilayer_perceptron_t));
    memset(mlp, 0, sizeof(*mlp));

    // MLP properties:
    mlp->epoch_count = epoch_count;
    mlp->input_count = p_input_count;

    // Perceptron properties:
    mlp->p_hidden1_count = p_hidden1_count;
    mlp->p_output_count = p_output_count;

    // Init the perceptrons:
    mlp->p_hidden1 = malloc(mlp->p_hidden1_count * sizeof(perceptron_t*));
    mlp->p_output = malloc(mlp->p_output_count * sizeof(perceptron_t*));

    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        mlp->p_hidden1[i] = init_perceptron(mlp->input_count, hidden1_activation_function, hidden1_derivative_activation_function, 1);
    }

    for (int i = 0; i < mlp->p_output_count; i++) {
        mlp->p_output[i] = init_perceptron(mlp->p_hidden1_count, output_activation_function, output_derivative_activation_function, 1);
    }

    // Init the arrays that hold the output of each stage (to be passed as input into the next stage):
    mlp->p_hidden1_output = (double *)malloc(sizeof(double) * mlp->p_hidden1_count);
    memset(mlp->p_hidden1_output, 0, sizeof(sizeof(double) * mlp->p_hidden1_count));

    mlp->p_output_output = (double *)malloc(sizeof(double) * mlp->p_output_count);
    memset(mlp->p_output_output, 0, sizeof(sizeof(double) * mlp->p_output_count));

    return mlp;
}

void destroy_mlp(multilayer_perceptron_t *mlp) {

    free(mlp->p_hidden1_output);
    free(mlp->p_output_output);

    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        destroy_perceptron(mlp->p_hidden1[i]);
    }
    free(mlp->p_hidden1);

    for (int i = 0; i < mlp->p_output_count; i++) {
        destroy_perceptron(mlp->p_output[i]);
    }
    free(mlp->p_output);
    
    free(mlp);
}

// Helper function to add binary classifier logic to an mlp:
double step_function(double x) {
    return x > 0.5 ? 1 : 0;
}

void mlp_feedforward(multilayer_perceptron_t *mlp, const double training_features[mlp->input_count]) {
    
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        // Pass in the complete set of training features as input to each perceptron in the hidden layer and capture the
        // activated output in the p_hidden1_output array:
        mlp->p_hidden1_output[k] = perceptron_feedforward(mlp->p_hidden1[k], training_features);
        // printf("\t\tHidden1[%d]: %f\n", k, mlp->p_hidden1_output[k]);
    }

    // Activate output layer:
    // For each perceptron in the output layer:
    for (int k = 0; k < mlp->p_output_count; k++) {
        // Pass in the output of the hidden layer as input to each perceptron in the output layer and capture the activated output:
        mlp->p_output_output[k] = perceptron_feedforward(mlp->p_output[k], mlp->p_hidden1_output);
        // printf("\t\tOutput[%d]: %f\n", k, mlp->p_output_output[k]);
    }
}

void mlp_backpropagate(multilayer_perceptron_t *mlp, const double training_features[], const double training_labels[], double learning_rate) {
    double output_error[mlp->p_output_count];
    double hidden1_error[mlp->p_hidden1_count];

    // Calculate the error for each perceptron in the output layer
    for (int k = 0; k < mlp->p_output_count; k++) {
        double output = mlp->p_output_output[k];
        output_error[k] = (output - training_labels[k]) * mlp->p_output[k]->derivative_activation_function(output);
    }

    // Calculate the error for each perceptron in the hidden layer
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        hidden1_error[k] = 0.0;
        for (int j = 0; j < mlp->p_output_count; j++) {
            hidden1_error[k] += output_error[j] * mlp->p_output[j]->weights[k];
        }
        double hidden_output = mlp->p_hidden1_output[k];
        hidden1_error[k] *= mlp->p_hidden1[k]->derivative_activation_function(hidden_output);
    }

    // Update the weights and biases for the output layer
    for (int k = 0; k < mlp->p_output_count; k++) {
        for (int j = 0; j < mlp->p_hidden1_count; j++) {
            mlp->p_output[k]->weights[j] -= learning_rate * output_error[k] * mlp->p_hidden1_output[j];
        }
        mlp->p_output[k]->bias_weight -= learning_rate * output_error[k];
    }

    // Update the weights and biases for the hidden layer
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        for (int j = 0; j < mlp->input_count; j++) {
            mlp->p_hidden1[k]->weights[j] -= learning_rate * hidden1_error[k] * training_features[j];
        }
        mlp->p_hidden1[k]->bias_weight -= learning_rate * hidden1_error[k];
    }
}

void train_mlp(multilayer_perceptron_t *mlp, int feature_count, int feature_dimension, const double training_features[feature_count][feature_dimension],
    int label_dimension, const double training_labels[feature_count][label_dimension], const double learning_rate) {

    // Exit if trying to train based on more features than expected:
    if (mlp->input_count != feature_dimension) {
        printf("Invalid Input\n");
        return;
    }

    // Foreach Epoch:
    for (int epoch = 0; epoch < mlp->epoch_count; epoch++) {
        // Foreach training vector:
        for (int i = 0; i < feature_count; i++) {

            // Debug print:
            // printf("Epoch: %d, Training Row: %d\n", epoch, i);
            // printf("\tfeature:");
            // for (int j = 0; j < column_count; j++) {
            //     printf("%f ", training_features[i][j]);
            // }
            // printf("label:%f\n", training_labels[i]);

            // Predict and train:
            mlp_feedforward(mlp, training_features[i]);
            mlp_backpropagate(mlp, training_features[i], training_labels[i], learning_rate);
        }
    }
}
