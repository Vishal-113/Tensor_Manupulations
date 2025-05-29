# TensorFlow Assignments: Tensors, Loss Functions & TensorBoard

This notebook demonstrates three key concepts in TensorFlow: tensor manipulation, loss function behavior, and training visualization with TensorBoard.

---

## 1. Tensor Manipulations & Reshaping

**Goal:** Learn how to create, reshape, and manipulate tensors using TensorFlow.

### Steps:
1. **Create a tensor** with shape `(4, 6)` filled with random values.
2. **Inspect the tensor**:
   - **Rank** is the number of dimensions (e.g., 2 for a matrix).
   - **Shape** gives the size along each axis.
3. **Reshape the tensor** to shape `(2, 3, 4)` to convert it into a 3D tensor.
4. **Transpose it** to shape `(3, 2, 4)` by reordering the axes.
5. **Broadcasting**:
   - A small tensor of shape `(1, 4)` is added to the larger tensor.
   - TensorFlow automatically expands dimensions to allow element-wise operations.

**Broadcasting Explanation:**
TensorFlow uses broadcasting to perform element-wise operations between tensors of different shapes by automatically expanding the smaller tensor without copying data.

---

## 2. Loss Functions & Hyperparameter Tuning

**Goal:** Understand and compare different loss functions by evaluating prediction changes.

### Steps:
1. Define **true labels** (`y_true`) and **model predictions** (`y_pred`).
2. Calculate:
   - **Mean Squared Error (MSE)**: Measures average squared difference between predictions and true values.
   - **Categorical Cross-Entropy (CCE)**: Used for classification, measures the difference between true labels and predicted probabilities.
3. Slightly change the predictions and observe how the loss values respond.
4. Plot the results using **Matplotlib** to compare the loss values across predictions.

---

## 3. Train a Neural Network and Log to TensorBoard

**Goal:** Train a neural network on MNIST and visualize training metrics with TensorBoard.

### Steps:
1. **Load and preprocess MNIST dataset**:
   - Normalize image pixel values to range `[0, 1]`.
2. **Define a simple neural network**:
   - Flatten layer, one hidden dense layer with ReLU, dropout for regularization, and output softmax layer.
3. **Train the model for 5 epochs** and enable TensorBoard logging using `TensorBoard` callback.
4. **View logs in TensorBoard**:
   - Launch with: `tensorboard --logdir=logs/fit`
   - Observe trends in **training and validation accuracy/loss**.

---

## TensorBoard Insights

1. **Training vs. Validation Accuracy**:
   - Training accuracy typically increases.
   - Validation accuracy may plateau or decrease (overfitting).

2. **Detecting Overfitting**:
   - If training accuracy continues rising but validation accuracy falls, the model is likely overfitting.

3. **Effect of More Epochs**:
   - May improve training accuracy but risk overfitting.
   - TensorBoard helps visualize when to stop training.

---

This notebook is useful for learning how TensorFlow handles tensors, evaluates model accuracy using different loss functions, and enables debugging and optimization via TensorBoard.
