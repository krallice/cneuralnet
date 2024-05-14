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
    mlp->p_hidden1_count = p_hidden1_count;
    mlp->p_output_count = p_output_count;

    // Set shaping (activation) functions:
    mlp->hidden1_activation_function = hidden1_activation_function;
    mlp->hidden1_derivative_activation_function = hidden1_derivative_activation_function;
    mlp->output_activation_function = output_activation_function;
    mlp->output_derivative_activation_function = output_derivative_activation_function;

    mlp->p_hidden1_weights = (double **)malloc(sizeof(double *) * mlp->input_count);
    for (int i = 0; i <p_input_count; i++) {
        mlp->p_hidden1_weights[i] = (double *)malloc(sizeof(double) * mlp->p_hidden1_count);
        for (int j = 0; j < mlp->p_hidden1_count; j++) {
            mlp->p_hidden1_weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; 
        }
    }

    mlp->p_output_weights = (double *)malloc(sizeof(double *) * mlp->p_hidden1_count);
    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        mlp->p_output_weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    return mlp;
}

void destroy_mlp(multilayer_perceptron_t *mlp) {

    // Free memory for weights
    for (int i = 0; i < mlp->input_count; i++) {
        free(mlp->p_hidden1_weights[i]);
    }
    free(mlp->p_hidden1_weights);
    free(mlp->p_output_weights);

    free(mlp);
}

void train_mlp(multilayer_perceptron_t *mlp, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate) {
    return;
}