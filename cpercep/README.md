```
# ./build.sh && ./main model_x_gt_9

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
```