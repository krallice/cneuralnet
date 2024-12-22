# Network Networks (In C)

This project is an attempt to document my understanding of the logic and mathematics behind neural networks, and potentially other ML techniques. To truly root my understanding of these systems, I've embarked upon the challenge to build/model them in C, using no external libraries like numpy or scikit. 

I've commited to only writing code if I can explain its reason for existing - no blindly copying or translating source code from other already written sources. The code is often written consulting the maths behind the algorithms, and I've taken lengths to ensure the code is adequately documented, including my thoughts as I go along. If I cannot explain the reason for a line of code, I don't write it until i can.

I started this project with the intent to remove the "magic" from deep learning, however as I've come to understand it, and the maths behind how deep learning models operate, I've only come to see them as more impressive and magic.

The aim isnt to build performant or perfect code. This is a live documentation of my learning processing.

**Architectures**

- Rosenblatt Perceptron
- Feed Forward Neural Network (1 Hidden Layer)
- *TODO:* Feed Forward Neural Network (2 Hidden Layers)

**Learning Methods**

- Stochastic Gradient Descent
- *TODO: Mini-Batch Gradient Descent*
- *TODO: Batch Gradient Descent*

## Models Implemented

**Perceptron/Single Neuron**
- model_x_gt_9 - A single dimensional input to a single perceptron, trained on the dataset of ```x > 9```
- model_linear - A two dimensional input perceptron, trained to model ```y = x/2 + 5```
- model_AND - A two dimensional input perceptron, trained to operate as an ```AND``` gate

**Feed Forward Neural Network (1 Hidden Layer)**
- model_4x2_mlp - FeedForward Neural Network (1-1-1) trained to learn the output of equation ```4 x 2```
- model_x2_mlp - FNN (1-1-1) trained to learn the equation ```y = 2x```
- model_x2plus1_mlp - FNN (1-1-1) trained to learn the equation ```y = 2x + 1```
- model_XOR - FNN modelling the ```XOR``` function
- model_2dout - FNN, outputing a 2d vector
- mnist_train - Train a FNN on the MNIST dataset
- mnist_test - Inference on a FNN for the MNIST dataset

**Feed Forward Neural Network (2 Hidden Layers)**
- model_x2plus1_2fnn - 2 Layer FNN (1-1-1-1) trained to learn the equation ```y = 2x + 1```
- mnist2_train - Train a FNN on the MNIST dataset
- mnist2_test - Inference on a FNN for the MNIST dataset

## Usage Example Dump

```
# ./main mnist_test

[ DETAILS ]
Model: mnist_test
Aim: Test a feed forward neural network on previously unseen MNIST dataset handwritten digits.
Architecture: 748 Input Nodes, 40 Hidden Nodes, 10 Output Nodes.
Hidden Activation: ReLU, Output Activation: ReLU
Loss Function: Mean Squared Error + Gradient Descent + Back Propagation 

Testing Size (n): 10000


[ LOADING WEIGHTS ]
Loading Model Weights from weights.bin


[ PREDICTION RESULTS ]
Testing set size: 10000
Success count: 8579
Success rate: 85.79%



# ./main model_x_gt_9

[ DETAILS ]
Model: model_x_gt_9
Aim: Train a neuron to fire when input vector x_1 > 9
Architecture: Single Perceptron
Input: A one dimensional input vector, x
        - x_1: Input value
Activation: Sign Activation Function
Loss Function: Perceptron Learning Rule
        (weight = weight + (learning_rate)(error := correct - predicted)(input))
Training Strategy:
        1000 epochs of correctly labeled integers 0 -> 19 inclusive. No rational/fractional numbers, integers only.
        Expected outcome is a normal vector that defines a hyperplane that seperates a two dimensional vector space based on the x value being < 9.


[ TRAINING ]
Model execution starting now ...
Training 1000 epochs now.
Training complete.


[ TRAINING RESULTS ]
Final weight vector w in R2 (w_0, w_1) = (0.440017, -4.173418)
Vector w is a normal vector to a hyperplane/vector subspace defined by: 0.44x + -4.17y = 0
The hyperplane crosses y = 1 at x = 9.48
        (y = 1 is important as all input vectors sit at (x=x,y=1) due to the bias term lifting the one dimensional input vectors consisting of x_0 into R2)
The perceptron will only fire for values that are >= 9.48


[ PREDICTION ]
Starting prediction test from x = -15 to 15
[ 01/30 SUCCESS ]: Input: -15 Expected: -1 Prediction: -1
[ 02/30 SUCCESS ]: Input: -14 Expected: -1 Prediction: -1
[ 03/30 SUCCESS ]: Input: -13 Expected: -1 Prediction: -1
[ 04/30 SUCCESS ]: Input: -12 Expected: -1 Prediction: -1
[ 05/30 SUCCESS ]: Input: -11 Expected: -1 Prediction: -1
[ 06/30 SUCCESS ]: Input: -10 Expected: -1 Prediction: -1
[ 07/30 SUCCESS ]: Input:  -9 Expected: -1 Prediction: -1
[ 08/30 SUCCESS ]: Input:  -8 Expected: -1 Prediction: -1
[ 09/30 SUCCESS ]: Input:  -7 Expected: -1 Prediction: -1
[ 10/30 SUCCESS ]: Input:  -6 Expected: -1 Prediction: -1
[ 11/30 SUCCESS ]: Input:  -5 Expected: -1 Prediction: -1
[ 12/30 SUCCESS ]: Input:  -4 Expected: -1 Prediction: -1
[ 13/30 SUCCESS ]: Input:  -3 Expected: -1 Prediction: -1
[ 14/30 SUCCESS ]: Input:  -2 Expected: -1 Prediction: -1
[ 15/30 SUCCESS ]: Input:  -1 Expected: -1 Prediction: -1
[ 16/30 SUCCESS ]: Input:   0 Expected: -1 Prediction: -1
[ 17/30 SUCCESS ]: Input:   1 Expected: -1 Prediction: -1
[ 18/30 SUCCESS ]: Input:   2 Expected: -1 Prediction: -1
[ 19/30 SUCCESS ]: Input:   3 Expected: -1 Prediction: -1
[ 20/30 SUCCESS ]: Input:   4 Expected: -1 Prediction: -1
[ 21/30 SUCCESS ]: Input:   5 Expected: -1 Prediction: -1
[ 22/30 SUCCESS ]: Input:   6 Expected: -1 Prediction: -1
[ 23/30 SUCCESS ]: Input:   7 Expected: -1 Prediction: -1
[ 24/30 SUCCESS ]: Input:   8 Expected: -1 Prediction: -1
[ 25/30 SUCCESS ]: Input:   9 Expected: -1 Prediction: -1
[ 26/30 SUCCESS ]: Input:  10 Expected:  1 Prediction:  1
[ 27/30 SUCCESS ]: Input:  11 Expected:  1 Prediction:  1
[ 28/30 SUCCESS ]: Input:  12 Expected:  1 Prediction:  1
[ 29/30 SUCCESS ]: Input:  13 Expected:  1 Prediction:  1
[ 30/30 SUCCESS ]: Input:  14 Expected:  1 Prediction:  1


# ./main model_linear

[ DETAILS ]
Model: model_linear
Aim: Train a neuron to linearly separate input vector x in R2 against equation: y = x/2 + 5
Architecture: Single Perceptron
Input: A two dimensional input vector, x
        - x_1: Input value, mapped conceptually to the x axis
        - x_2: Input value, mapped conceptually to the y axis
Activation: Sign Activation Function
Loss Function: Perceptron Learning Rule
        (weight = weight + (learning_rate)(error := correct - predicted)(input))
Training Strategy:
        Training data randomly generated of 50 entries, with domain: (-20 <= x <= 20) and range (-20 <= x <= 20)
        Perceptron trained with 100 epochs of the dataset.


[ GENERATE TRAINING DATA ]
Generated Random Point (1/50): (x,y): (-7, 2). Point is above the line.
Generated Random Point (2/50): (x,y): (18, 20). Point is above the line.
Generated Random Point (3/50): (x,y): (17, -13). Point is below the line.
Generated Random Point (4/50): (x,y): (1, 18). Point is above the line.
Generated Random Point (5/50): (x,y): (4, -11). Point is below the line.
Generated Random Point (6/50): (x,y): (-7, 5). Point is above the line.
Generated Random Point (7/50): (x,y): (0, -6). Point is below the line.
Generated Random Point (8/50): (x,y): (-14, 10). Point is above the line.
Generated Random Point (9/50): (x,y): (-4, -8). Point is below the line.
Generated Random Point (10/50): (x,y): (-20, -12). Point is below the line.
Generated Random Point (11/50): (x,y): (-19, -18). Point is below the line.
Generated Random Point (12/50): (x,y): (-17, -3). Point is above the line.
Generated Random Point (13/50): (x,y): (-20, -20). Point is below the line.
Generated Random Point (14/50): (x,y): (-13, 18). Point is above the line.
Generated Random Point (15/50): (x,y): (7, 14). Point is above the line.
Generated Random Point (16/50): (x,y): (-8, 20). Point is above the line.
Generated Random Point (17/50): (x,y): (-5, -11). Point is below the line.
Generated Random Point (18/50): (x,y): (-20, -7). Point is below the line.
Generated Random Point (19/50): (x,y): (-1, 4). Point is above the line.
Generated Random Point (20/50): (x,y): (-8, -18). Point is below the line.
Generated Random Point (21/50): (x,y): (13, 7). Point is below the line.
Generated Random Point (22/50): (x,y): (8, -8). Point is below the line.
Generated Random Point (23/50): (x,y): (-20, 14). Point is above the line.
Generated Random Point (24/50): (x,y): (-18, -3). Point is above the line.
Generated Random Point (25/50): (x,y): (-12, -16). Point is below the line.
Generated Random Point (26/50): (x,y): (5, -11). Point is below the line.
Generated Random Point (27/50): (x,y): (-11, 10). Point is above the line.
Generated Random Point (28/50): (x,y): (6, -11). Point is below the line.
Generated Random Point (29/50): (x,y): (10, 15). Point is above the line.
Generated Random Point (30/50): (x,y): (-12, -4). Point is below the line.
Generated Random Point (31/50): (x,y): (8, 3). Point is below the line.
Generated Random Point (32/50): (x,y): (-3, -16). Point is below the line.
Generated Random Point (33/50): (x,y): (12, 0). Point is below the line.
Generated Random Point (34/50): (x,y): (-1, -10). Point is below the line.
Generated Random Point (35/50): (x,y): (-17, 11). Point is above the line.
Generated Random Point (36/50): (x,y): (-5, 18). Point is above the line.
Generated Random Point (37/50): (x,y): (0, -18). Point is below the line.
Generated Random Point (38/50): (x,y): (-8, 0). Point is below the line.
Generated Random Point (39/50): (x,y): (18, -6). Point is below the line.
Generated Random Point (40/50): (x,y): (19, -15). Point is below the line.
Generated Random Point (41/50): (x,y): (0, 5). Point is above the line.
Generated Random Point (42/50): (x,y): (-4, 9). Point is above the line.
Generated Random Point (43/50): (x,y): (-3, -16). Point is below the line.
Generated Random Point (44/50): (x,y): (-20, -14). Point is below the line.
Generated Random Point (45/50): (x,y): (19, -12). Point is below the line.
Generated Random Point (46/50): (x,y): (5, 9). Point is above the line.
Generated Random Point (47/50): (x,y): (11, -17). Point is below the line.
Generated Random Point (48/50): (x,y): (15, 5). Point is below the line.
Generated Random Point (49/50): (x,y): (5, -6). Point is below the line.
Generated Random Point (50/50): (x,y): (17, 10). Point is below the line.


[ TRAINING ]
Model execution starting now ...
Training 100 epochs now.


[ TRAINING RESULTS ]
Final weight vector w in R2 (w_0, w_1) = (-3.23, 8.04)
Final bias value b = -28.65
This defines an equation: -3.23x + 8.04y + -28.65 = 0
Re-arranged for y: y = 0.40x + 3.56


[ PREDICTION ]
[ 01/20 SUCCESS ]: Input: (  8, -5) Expected: -1 (Point is below the line) Prediction: -1
[ 02/20 SUCCESS ]: Input: ( 20,-10) Expected: -1 (Point is below the line) Prediction: -1
[ 03/20 SUCCESS ]: Input: ( -7, -5) Expected: -1 (Point is below the line) Prediction: -1
[ 04/20 SUCCESS ]: Input: (-12,  0) Expected:  1 (Point is above the line) Prediction:  1
[ 05/20 SUCCESS ]: Input: ( 15, 14) Expected:  1 (Point is above the line) Prediction:  1
[ 06/20 SUCCESS ]: Input: ( 17,  4) Expected: -1 (Point is below the line) Prediction: -1
[ 07/20 SUCCESS ]: Input: (-10,-20) Expected: -1 (Point is below the line) Prediction: -1
[ 08/20 SUCCESS ]: Input: (  4, -2) Expected: -1 (Point is below the line) Prediction: -1
[ 09/20 SUCCESS ]: Input: (-20, 14) Expected:  1 (Point is above the line) Prediction:  1
[ 10/20 SUCCESS ]: Input: (-16, 11) Expected:  1 (Point is above the line) Prediction:  1
[ 11/20 SUCCESS ]: Input: (  7,-10) Expected: -1 (Point is below the line) Prediction: -1
[ 12/20 SUCCESS ]: Input: (  6, -9) Expected: -1 (Point is below the line) Prediction: -1
[ 13/20 SUCCESS ]: Input: ( 17,-19) Expected: -1 (Point is below the line) Prediction: -1
[ 14/20 SUCCESS ]: Input: (-13,  7) Expected:  1 (Point is above the line) Prediction:  1
[ 15/20 SUCCESS ]: Input: (-11,  1) Expected:  1 (Point is above the line) Prediction:  1
[ 16/20 SUCCESS ]: Input: ( -6, 18) Expected:  1 (Point is above the line) Prediction:  1
[ 17/20 SUCCESS ]: Input: ( 16, -5) Expected: -1 (Point is below the line) Prediction: -1
[ 18/20 SUCCESS ]: Input: (-13,-12) Expected: -1 (Point is below the line) Prediction: -1
[ 19/20 SUCCESS ]: Input: ( 10, -3) Expected: -1 (Point is below the line) Prediction: -1
[ 20/20 SUCCESS ]: Input: ( 11,  6) Expected: -1 (Point is below the line) Prediction: -1

```
