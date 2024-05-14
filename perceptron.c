#include "perceptron.h"

// ////////////////////////////////////  //
//            Create/Destroy             //
//  ///////////////////////////////////  //

perceptron_t *init_perceptron(const int input_count, double (*activation_function)(double), double (*derivative_activation_function)(double), int training_epoch_count) {
    
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
    p->derivative_activation_function = derivative_activation_function;
    p->training_epoch_count = training_epoch_count;

    return p;
}

void destroy_perceptron(perceptron_t *p) {
    free(p->weights);
    free(p);
    return;
}

// ////////////////////////////////////  //
//     Non-Differentiable Activation     //
//               Functions               //
//  ///////////////////////////////////  //

// These activation functions are not differentiable, and are used for the perceptron model,
// which consists of a single node; therefore backpropegation is not required (which relies on the derivative of the activation function to calculate the gradient of the loss function).

double sign_activation_function(double x) {
    return x < 0 ? -1 : 1;
}

// ////////////////////////////////////  //
//         Differentiable Activation     //
//               Functions               //
//  ///////////////////////////////////  //

// The following activation functions are used in neural networks, where backpropegation is required to calculate the gradient of the loss function.

double linear_activation(double x) {
    return x;
}

double derivative_linear_activation(double x) {
    return 1;
}

// Rectified Linear Unit:
// Nice properties from it being close to linear (TODO: describe why later)
double relu_activation(double x) {
    return x > 0 ? x : 0;
}

double derivative_relu_activation(double x) {
    return x > 0 ? 1 : 0;
}

// Sigmoid
// TOOD: Describe properties later

double sigmoid_activation(double x) {
    return 1 / (1 + exp(-x));
}

double derivative_sigmoid_activation(double x) {
    return x * (1 - x);
}

// ////////////////////////////////////  //
//               Predict                 //
//  ///////////////////////////////////  //

double perceptron_feedforward(perceptron_t *p, const double training_features[]) {

    // Perform a weighted sum of all training features, and run it through the activation function to produce a final prediction output.

    // Algebraically, given:
    // w = vector of weights
    // x = vector of features
    // act() = activation function
    // This function models the following equation:
    // y = act(f(x)) = act(w.x + b) OR act(w.x) (if the bias term 1 is bundled into x)
    
    // The weight vector is orthogonal (indicates the normal) to a hyperplane that we are searching for, that would seperate the dataset into 2 linearly seperated binary classes.
    // Performing the dot product between w and x determines which 'side' of the hyperplane a particular data vector x lies:
    // if >0, it lies on the side of the hyperplane that the normal points to; <0, it points away

    // This output is then sent to the activation function to be squished/transformed before the loss is calculated

    // On the bias, there are two geometric ways to reason with it:

    // Parameter space (including the bias term into the data vector x): This shifts the dataset up by dimension which allows the vector subspace (defined by the hyperplane) -
    // which by very definition of a subspace, MUST cross the origin; to be able to linearly split all datasets.
    // For example, a vector subspace crossing the origin could not split the two x vectors (1) and (3), as they sit directly on the X-axis.
    // It could however, split this data if it were raised up a dimension by including the bias into x: (1,1) and (3, 1).

    // Feature space (Adding the bias as an additional term to make x.w + b): This shifts the intercept of the hyperplane so that it has a degree of freedom. A hyperplane free
    // from the constraint of requiring origin intersection, can now separate (1) and (3).

    double weighted_sum = p->bias_weight;
    for (int i = 0; i < p->input_count; i++) {
        weighted_sum += training_features[i] * p->weights[i];
    }
    return p->activation_function(weighted_sum);
}

// ////////////////////////////////////  //
//            Training Loop              //
//  ///////////////////////////////////  //

void train_perceptron(perceptron_t *p, int row_count, int column_count, const double training_features[row_count][column_count], const double training_labels[row_count], const double learning_rate) {

    // Exit if trying to train based on more features than expected:
    if (p->input_count != column_count) {
        printf("Invalid Input\n");
        return;
    }

    // Iterate over the dataset training_epoch_count times:
    for (int epoch = 0; epoch < p->training_epoch_count; epoch++) {
        
        // Foreach entry in the total dataset:
        for (int i = 0; i < row_count; i++) {

            // Implement the Rosenblatt Perceptron Learning Rule:
            // error = target - guess
            // new_weight = current_weight + (error)(input)(learning_rate)

            // Create a prediction for this particular vector of features:
            double prediction = perceptron_feedforward(p, training_features[i]);

            // First calculate the error difference (Difference in expected/predicted values).
            // If the error_difference == 0, then the weights are effectively not adjusted as the prediction was correct (there's nothing to do!)
            // If the error_difference is non-zero, then we were incorrect in some direction, and we need to adjust accordingly
            // Predicted    Label   Error
            //  1            1       0 (correct response)
            //  1           -1      -2 (incorrect, was a false positive)
            // -1            1       2 (incorrect. was a false negative)
            // -1           -1       0 (correct response)
            double error_difference = training_labels[i] - prediction;

            // Given the error_difference, calculate:
            // new_weight = current_weight + (error)(input)(learning_rate)

            // learning_rate = hyperparameter as a magic scalar that scales how drastically the weight vector is adjusted.
            // larger learning_rates have certain pros:
            //  + Faster Convergence:
            //      With a large learning rate, the model may converge faster since it takes larger steps towards the optimal weights. 
            //      This can be advantageous, especially when training large datasets.
            //  + Escape Local Minima: 
            //      A larger learning rate may help the model escape from local minima in the loss function. 
            //      It allows the model to jump out of regions where the gradient is small and move towards the global minimum.
            // and cons:
            //  - Overshooting: 
            //      There's a risk of overshooting the minimum. 
            //      If the learning rate is too large, the algorithm might oscillate or diverge, missing the minimum point and failing to converge.
            //  - Instability: 
            //      High learning rates can lead to instability, making the model's behavior sensitive to small changes in the input data or the initial weights.
            
            // First address the bias (same formula, with training feature implicit as 1), then nudge the rest of the weight vector entries:
            p->bias_weight += learning_rate * error_difference;
            for (int j = 0; j < p->input_count; j++) {
                p->weights[j] += learning_rate * error_difference * training_features[i][j];
            }
        }
    }

    return;
}