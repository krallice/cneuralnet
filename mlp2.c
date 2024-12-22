#include "mlp2.h"

multilayer_perceptron2_t *init_mlp2(int p_input_count, int p_hidden1_count, int p_hidden2_count, int p_output_count, 
    double (*hidden1_activation_function)(double), double (*hidden1_derivative_activation_function)(double), 
    double (*hidden2_activation_function)(double), double (*hidden2_derivative_activation_function)(double), 
    double (*output_activation_function)(double),  double (*output_derivative_activation_function)(double), int epoch_count) {

    multilayer_perceptron2_t *mlp = (multilayer_perceptron2_t*)malloc(sizeof(multilayer_perceptron2_t));
    memset(mlp, 0, sizeof(*mlp));

    mlp->epoch_count = epoch_count;
    mlp->input_count = p_input_count;

    mlp->p_hidden1_count = p_hidden1_count;
    mlp->p_hidden2_count = p_hidden2_count;
    mlp->p_output_count = p_output_count;

    mlp->p_hidden1 = malloc(mlp->p_hidden1_count * sizeof(perceptron_t*));
    mlp->p_hidden2 = malloc(mlp->p_hidden2_count * sizeof(perceptron_t*));
    mlp->p_output = malloc(mlp->p_output_count * sizeof(perceptron_t*));

    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        mlp->p_hidden1[i] = init_perceptron(mlp->input_count, hidden1_activation_function, hidden1_derivative_activation_function, 1);
    }

    for (int i = 0; i < mlp->p_hidden2_count; i++) {
        mlp->p_hidden2[i] = init_perceptron(mlp->p_hidden1_count, hidden2_activation_function, hidden2_derivative_activation_function, 1);
    }

    for (int i = 0; i < mlp->p_output_count; i++) {
        mlp->p_output[i] = init_perceptron(mlp->p_hidden2_count, output_activation_function, output_derivative_activation_function, 1);
    }

    mlp->p_hidden1_output = (double *)malloc(sizeof(double) * mlp->p_hidden1_count);
    memset(mlp->p_hidden1_output, 0, sizeof(double) * mlp->p_hidden1_count);

    mlp->p_hidden2_output = (double *)malloc(sizeof(double) * mlp->p_hidden2_count);
    memset(mlp->p_hidden2_output, 0, sizeof(double) * mlp->p_hidden2_count);

    mlp->p_output_output = (double *)malloc(sizeof(double) * mlp->p_output_count);
    memset(mlp->p_output_output, 0, sizeof(double) * mlp->p_output_count);

    return mlp;
}

void destroy_mlp2(multilayer_perceptron2_t *mlp) {

    free(mlp->p_hidden1_output);
    free(mlp->p_hidden2_output);
    free(mlp->p_output_output);

    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        destroy_perceptron(mlp->p_hidden1[i]);
    }
    free(mlp->p_hidden1);

    for (int i = 0; i < mlp->p_hidden2_count; i++) {
        destroy_perceptron(mlp->p_hidden2[i]);
    }
    free(mlp->p_hidden2);

    for (int i = 0; i < mlp->p_output_count; i++) {
        destroy_perceptron(mlp->p_output[i]);
    }
    free(mlp->p_output);
    
    free(mlp);
}

void mlp2_feedforward(multilayer_perceptron2_t *mlp, const double training_features[mlp->input_count]) {

    // Feedforward the activations through the network:
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        mlp->p_hidden1_output[k] = perceptron_feedforward(mlp->p_hidden1[k], training_features);
    }

    for (int k = 0; k < mlp->p_hidden2_count; k++) {
        mlp->p_hidden2_output[k] = perceptron_feedforward(mlp->p_hidden2[k], mlp->p_hidden1_output);
    }

    for (int k = 0; k < mlp->p_output_count; k++) {
        mlp->p_output_output[k] = perceptron_feedforward(mlp->p_output[k], mlp->p_hidden2_output);
    }
}

void mlp2_backpropagate(multilayer_perceptron2_t *mlp, const double training_features[], const double training_labels[], double learning_rate) {

    double output_dLdz[mlp->p_output_count];
    double hidden2_dLdz[mlp->p_hidden2_count];
    double hidden1_dLdz[mlp->p_hidden1_count];

    // Please see mlp.c for the backpropagation/gradient descent equations

    // Stage 1. Calculate the gradient of the loss function with respect to z (the pre-activated output of the node):

    for (int k = 0; k < mlp->p_output_count; k++) {
        output_dLdz[k] = (mlp->p_output_output[k] - training_labels[k]) * mlp->p_output[k]->derivative_activation_function(mlp->p_output_output[k]);
    }

    for (int k = 0; k < mlp->p_hidden2_count; k++) {
        hidden2_dLdz[k] = 0.0;
        for (int j = 0; j < mlp->p_output_count; j++) {
            hidden2_dLdz[k] += mlp->p_output[j]->weights[k] * output_dLdz[j];
        }
        hidden2_dLdz[k] *= mlp->p_hidden2[k]->derivative_activation_function(mlp->p_hidden2_output[k]);
    }

    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        hidden1_dLdz[k] = 0.0;
        for (int j = 0; j < mlp->p_hidden2_count; j++) {
            hidden1_dLdz[k] += mlp->p_hidden2[j]->weights[k] * hidden2_dLdz[j];
        }
        hidden1_dLdz[k] *= mlp->p_hidden1[k]->derivative_activation_function(mlp->p_hidden1_output[k]);
    }

    // Stage 2. Calculate the gradient of the loss function with respect to w (the input weight),
    // and update the weights using the update rule (gradient descent): w ← w - (α * (dL/dw))
    
    for (int k = 0; k < mlp->p_output_count; k++) {
        for (int j = 0; j < mlp->p_hidden2_count; j++) {
            mlp->p_output[k]->weights[j] -= learning_rate * (mlp->p_hidden2_output[j] * output_dLdz[k]);
        }
        mlp->p_output[k]->bias_weight -= learning_rate * output_dLdz[k];
    }

    for (int k = 0; k < mlp->p_hidden2_count; k++) {
        for (int j = 0; j < mlp->p_hidden1_count; j++) {
            mlp->p_hidden2[k]->weights[j] -= learning_rate * (mlp->p_hidden1_output[j] * hidden2_dLdz[k]);
        }
        mlp->p_hidden2[k]->bias_weight -= learning_rate * hidden2_dLdz[k];
    }

    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        for (int j = 0; j < mlp->input_count; j++) {
            mlp->p_hidden1[k]->weights[j] -= learning_rate * (training_features[j] * hidden1_dLdz[k]);
        }
        mlp->p_hidden1[k]->bias_weight -= learning_rate * hidden1_dLdz[k];
    }
}

void train_mlp2(multilayer_perceptron2_t *mlp, int feature_count, int feature_dimension, const double training_features[feature_count][feature_dimension],
    int label_dimension, const double training_labels[feature_count][label_dimension], const double learning_rate) {

    if (mlp->input_count != feature_dimension || mlp->p_output_count != label_dimension) {
        printf("Invalid Feature or Label Dimensionality.\n");
        return;
    }

    for (int epoch = 0; epoch < mlp->epoch_count; epoch++) {
        for (int i = 0; i < feature_count; i++) {
            mlp2_feedforward(mlp, training_features[i]);
            mlp2_backpropagate(mlp, training_features[i], training_labels[i], learning_rate);
        }
    }
}

void save_mlp2_weights(const multilayer_perceptron2_t *mlp, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for saving weights");
        return;
    }

    // Save the MLP structure
    fwrite(&mlp->input_count, sizeof(int), 1, file);
    fwrite(&mlp->p_hidden1_count, sizeof(int), 1, file);
    fwrite(&mlp->p_hidden2_count, sizeof(int), 1, file);
    fwrite(&mlp->p_output_count, sizeof(int), 1, file);

    // Save weights and biases for hidden1 layer
    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        fwrite(mlp->p_hidden1[i]->weights, sizeof(double), mlp->input_count, file);
        fwrite(&mlp->p_hidden1[i]->bias_weight, sizeof(double), 1, file);
    }

    // Save weights and biases for hidden2 layer
    for (int i = 0; i < mlp->p_hidden2_count; i++) {
        fwrite(mlp->p_hidden2[i]->weights, sizeof(double), mlp->p_hidden1_count, file);
        fwrite(&mlp->p_hidden2[i]->bias_weight, sizeof(double), 1, file);
    }

    // Save weights and biases for output layer
    for (int i = 0; i < mlp->p_output_count; i++) {
        fwrite(mlp->p_output[i]->weights, sizeof(double), mlp->p_hidden2_count, file);
        fwrite(&mlp->p_output[i]->bias_weight, sizeof(double), 1, file);
    }

    fclose(file);
}

void load_mlp2_weights(multilayer_perceptron2_t *mlp, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for loading weights");
        return;
    }

    int input_count, hidden1_count, hidden2_count, output_count;

    // Read and validate structure
    fread(&input_count, sizeof(int), 1, file);
    fread(&hidden1_count, sizeof(int), 1, file);
    fread(&hidden2_count, sizeof(int), 1, file);
    fread(&output_count, sizeof(int), 1, file);

    if (input_count != mlp->input_count || hidden1_count != mlp->p_hidden1_count ||
        hidden2_count != mlp->p_hidden2_count || output_count != mlp->p_output_count) {
        fclose(file);
        fprintf(stderr, "MLP structure does not match file contents\n");
        return;
    }

    // Load weights and biases for hidden1 layer
    for (int i = 0; i < mlp->p_hidden1_count; i++) {
        fread(mlp->p_hidden1[i]->weights, sizeof(double), mlp->input_count, file);
        fread(&mlp->p_hidden1[i]->bias_weight, sizeof(double), 1, file);
    }

    // Load weights and biases for hidden2 layer
    for (int i = 0; i < mlp->p_hidden2_count; i++) {
        fread(mlp->p_hidden2[i]->weights, sizeof(double), mlp->p_hidden1_count, file);
        fread(&mlp->p_hidden2[i]->bias_weight, sizeof(double), 1, file);
    }

    // Load weights and biases for output layer
    for (int i = 0; i < mlp->p_output_count; i++) {
        fread(mlp->p_output[i]->weights, sizeof(double), mlp->p_hidden2_count, file);
        fread(&mlp->p_output[i]->bias_weight, sizeof(double), 1, file);
    }

    fclose(file);
}
