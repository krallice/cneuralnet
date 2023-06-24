#include "mlp.h"

multilayer_perceptron_t *init_mlp(int p_input_count, int p_hidden1_count, int p_output_count, int epoch_count) {

    // Init and zeroise:
    multilayer_perceptron_t *mlp = (multilayer_perceptron_t*)malloc(sizeof(multilayer_perceptron_t));
    memset(mlp, 0, sizeof(*mlp));

    // MLP properties:
    mlp->epoch_count = epoch_count;
    mlp->input_count = p_input_count;

    // Perceptron properties:
    mlp->p_input_count = p_input_count;
    mlp->p_hidden1_count = p_hidden1_count;
    mlp->p_output_count = p_output_count;

    // Init the perceptrons:
    mlp->p_input = malloc(mlp->p_input_count * sizeof(perceptron_t*));
    mlp->p_hidden1 = malloc(mlp->p_hidden1_count * sizeof(perceptron_t*));
    mlp->p_output = malloc(mlp->p_output_count * sizeof(perceptron_t*));

    for (int i = 0; i < mlp->p_input_count; i++) {
        mlp->p_input[i] = init_perceptron(1, sign_activation_function, 1);
    }

    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        mlp->p_hidden1[i] = init_perceptron(mlp->p_input_count, sign_activation_function, 1);
    }

    for (int i = 0; i < mlp->p_output_count; i++) {
        mlp->p_output[i] = init_perceptron(mlp->p_hidden1_count, sign_activation_function, 1);
    }

    // Init the arrays that hold the output of each stage (to be passed as input into the next stage):
    mlp->p_input_output = (double *)malloc(sizeof(double) * mlp->p_input_count);
    memset(mlp->p_input_output, 0, sizeof(sizeof(double) * mlp->p_input_count));

    mlp->p_hidden1_output = (double *)malloc(sizeof(double) * mlp->p_hidden1_count);
    memset(mlp->p_hidden1_output, 0, sizeof(sizeof(double) * mlp->p_hidden1_count));

    mlp->p_output_output = (double *)malloc(sizeof(double) * mlp->p_output_count);
    memset(mlp->p_output_output, 0, sizeof(sizeof(double) * mlp->p_output_count));

    return mlp;
}

void destroy_mlp(multilayer_perceptron_t *mlp) {

    free(mlp->p_input_output);
    free(mlp->p_hidden1_output);
    free(mlp->p_output_output);
    
    for (int i = 0; i < mlp->p_input_count; i++) {
        destroy_perceptron(mlp->p_input[i]);
    }
    free(mlp->p_input);

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

void train_mlp(multilayer_perceptron_t *mlp, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate) {

    // Exit if trying to train based on more features than expected:
    if (mlp->input_count != column_count) {
        printf("Invalid Input\n");
        return;
    }

    for (int epoch = 0; epoch < mlp->epoch_count; epoch++) {
        for (int i = 0; i < row_count; i++) {
            printf("Epoch: %d, Training Row: %d\n", epoch, i);

            // Input layer:
            for (int j = 0; j < column_count; j++) {
                
            }
        }
    }
}