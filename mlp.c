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

void train_mlp(multilayer_perceptron_t *mlp, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate) {

    // Exit if trying to train based on more features than expected:
    if (mlp->input_count != column_count) {
        printf("Invalid Input\n");
        return;
    }

    // Foreach Epoch:
    // for (int epoch = 0; epoch < mlp->epoch_count; epoch++) {
    for (int epoch = 0; epoch < 1; epoch++) {
        // Foreach training vector:
        for (int i = 0; i < row_count; i++) {
            printf("Epoch: %d, Training Row: %d\n", epoch, i);
            
            // Debug print:
            printf("\tfeature:");
            for (int j = 0; j < column_count; j++) {
                printf("%f ", training_features[i][j]);
            }
            printf("label:%f\n", training_labels[i]);

            // Activate hidden layer:
            for (int k = 0; k < mlp->p_hidden1_count; k++) {
                mlp->p_hidden1_output[k] = perceptron_feedforward(mlp->p_hidden1[k], training_features[i]);
                printf("\t\tHidden1[%d]: %f\n", k, mlp->p_hidden1_output[k]);
            }

            // Activate output layer with the hidden layer as input:
            for (int k = 0; k < mlp->p_output_count; k++) {
                mlp->p_output_output[k] = perceptron_feedforward(mlp->p_output[k], mlp->p_hidden1_output);
                printf("\t\tOutput[%d]: %f\n", k, mlp->p_output_output[k]);
                // printf("\t\tOutput[%d] Step: %f\n", k, step_function(mlp->p_output_output[k]));
            }

            // Backpropagation
            // Update weights for each output node:
            for (int k = 0; k < mlp->p_output_count; k++) {
                double error = mlp->p_output_output[k] - training_labels[i];
                printf("ERROR: %f\n", error);
            }

        }
    }
}