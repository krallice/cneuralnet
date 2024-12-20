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
    
    // Activate hidden layer:
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        // Pass in the complete set of training features as input to each perceptron in the hidden layer and capture the activated output
        mlp->p_hidden1_output[k] = perceptron_feedforward(mlp->p_hidden1[k], training_features);
        printf("\t\tHidden1[%d]: %f\n", k, mlp->p_hidden1_output[k]);
    }

    // Activate output layer:
    for (int k = 0; k < mlp->p_output_count; k++) {
        // Pass in the output of the hidden layer as input to each perceptron in the output layer and capture the activated output:
        mlp->p_output_output[k] = perceptron_feedforward(mlp->p_output[k], mlp->p_hidden1_output);
        printf("\t\tOutput[%d]: %f\n", k, mlp->p_output_output[k]);
    }
}

void mlp_backpropagate(multilayer_perceptron_t *mlp, const double training_features[], const double training_labels[], double learning_rate) {
    
    // Equations of a node:
    // Pre-Activation: z = w * x + b
    // Activated: a = f(z), where f is the activation function

    // Back propegation 'unwinds' the above relationships to work out how much each weight should be adjusted to reduce the loss function to its minimum.
    // In this instance, we're using the mean squared error loss function (MSE): L = 1/2 * (a - y)^2
    // Technically, the definition of the MSE loss function is L = (y - a)^2, but the 1/2 is added as it 
    // doesn't affect the gradient of the loss function, but simplifies its calculation.

    double output_dLdz[mlp->p_output_count];
    double hidden1_dLdz[mlp->p_hidden1_count];

    // Stage 1. Calculate the gradient of the loss function with respect to z (the pre-activated output of the node):

    // For each node in the output layer, calculate dL/dz:
    for (int k = 0; k < mlp->p_output_count; k++) {
        // dL/dz = da/dz * dL/da
        // dL/dz =  f'(z) * (y - a)
        output_dLdz[k] = (mlp->p_output_output[k] - training_labels[k]) * mlp->p_output[k]->derivative_activation_function(mlp->p_output_output[k]);
        printf("output_dLdz[%d] = %f\n", k, output_dLdz[k]);
    }

    // For each node in the hidden1 layer, calculate dL/dz:
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        hidden1_dLdz[k] = 0.0;
        for (int j = 0; j < mlp->p_output_count; j++) {
            // Build up the sigma component of equation:
            // dL/dz = da/dz * Σ(dz/da * output_dLdz)
            // dL/dz = da/dz * Σ(w * output_dLdz)
            hidden1_dLdz[k] += mlp->p_output[j]->weights[k] * output_dLdz[j];
        }
        // Compute da/dz to finalise calculation of hidden1_dLdz:
        // dL/dz = f'(z) * Σ(w * output_dLdz)
        hidden1_dLdz[k] *= mlp->p_hidden1[k]->derivative_activation_function(mlp->p_hidden1_output[k]);
    }

    // Stage 2. Calculate the gradient of the loss function with respect to w (the input weight),
    // and update the weights using the update rule (gradient descent): w ← w - (α * (dL/dw))

    // Output layer:
    for (int k = 0; k < mlp->p_output_count; k++) {
        printf("Before update: output weights[%d][0] = %f\n", k, mlp->p_output[k]->weights[0]);
        for (int j = 0; j < mlp->p_hidden1_count; j++) {
            // Gradient descent for each weight associated with the output node,
            // noting that dL/dw = dz/dw * dL/dz, and dz/dw = a (the output of the hidden layer):
            // w ← w - (α * (dz/dw * dL/dz))
            mlp->p_output[k]->weights[j] -= learning_rate * (mlp->p_hidden1_output[j] * output_dLdz[k]);
        }
        printf("After update: output weights[%d][0] = %f\n", k, mlp->p_output[k]->weights[0]);
        // For the bias, dz/dw = 1, so we can simplify the equation to:
        // w ← w - (α * (1 * dL/dz))
        mlp->p_output[k]->bias_weight -= learning_rate * output_dLdz[k];
    }

    // Hidden layer, same logic as above:
    for (int k = 0; k < mlp->p_hidden1_count; k++) {
        for (int j = 0; j < mlp->input_count; j++) {
            mlp->p_hidden1[k]->weights[j] -= learning_rate * (training_features[j] * hidden1_dLdz[k]);
        }
        mlp->p_hidden1[k]->bias_weight -= learning_rate * hidden1_dLdz[k];
    }
}

void train_mlp(multilayer_perceptron_t *mlp, int feature_count, int feature_dimension, const double training_features[feature_count][feature_dimension],
    int label_dimension, const double training_labels[feature_count][label_dimension], const double learning_rate) {

    // Exit if trying to train based on more features than expected:
    if (mlp->input_count != feature_dimension) {
        printf("Invalid Feature Dimensionality.\n");
        return;
    }

    if (mlp->p_output_count != label_dimension) {
        printf("Invalid Label Dimensionality.\n");
        return;
    }

    // Foreach Epoch:
    for (int epoch = 0; epoch < mlp->epoch_count; epoch++) {
        // Foreach training vector:
        for (int i = 0; i < feature_count; i++) {

            // Debug print:
            printf("Epoch: %d, Training Row: %d\n", epoch, i);
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
