# A Backpropagation Alternative

## Motivation

In classical backpropagation, each weight and activation is updated via local gradient steps. However, when analyzing a **single neuron's** finite update step, we notice:

- Let `<x, w>` be the forward pass (dot product),
- `y` the actual output,
- `y'` the target output, and
- `dy = y' - y` the learning signal.

A small learning rate (`lr`) update in classical backprop for weights `w` and activations `x` (with a simplified factor `lr` applied symmetrically) would look like:

```
<x + dy * w * lr, w + dy * x * lr>
```


which expands approximately to:

```
<x, w> + dy * lr * (<x, x> + <w, w>) + (dy)^2 * lr^2 * <w, x>
```


This highlights that even a single finite-step update introduces quadratic terms and “length noise” from `<w, w>` and `<x, x>`. This explains why **small** and **adaptive** learning rates are often critical: large learning rates otherwise amplify unwanted terms.

In an **ideal** scenario, one might want `w'` and `x'` such that:

```
<w', x'> = y + dy * lr
```


implying a more direct adjustment to match the target.

## Key Insight

A single data point can be satisfied by infinitely many configurations of `(w, x)`. The real challenge is ensuring the **entire dataset** can be fit. The new method works by **backpropagating target activations** directly, reminiscent of a **conjugate-gradient**-like approach:

1. We choose **random directions** in the input/weight space of each layer.
2. We solve for how far to move along those directions so that the error is reduced in a batch-averaged sense.
3. We then propagate updated **activation targets** backward through the network.

Below is a step-by-step outline of the algorithm for one linear layer (and its associated activation).

---

## Algorithm Outline

### 1. Random Direction Sampling

- For a given linear layer, let its input dimension be `D`.
- **Sample** a random vector `d` in `R^D`.

### 2. Forward Pass and Error Computation

- Perform a forward pass on a **batch** of training data to obtain outputs `y`.
- Compute the error signals `dy = y' - y` for each sample in the batch, where `y'` is the target output.

### 3. Weight Update Along `d`

- For **each neuron** in this layer (with weights `w`):
  1. For each sample `i` in the batch, compute a **factor**:

     ```
     f_i = dy_i / <w, d>
     ```

  2. **Average** these factors across all samples in the batch:

     ```
     f_bar = (1 / N) * sum(f_i for i in range(N))
     ```

  3. Update the neuron’s weights **along** `d` using `f_bar`:

     ```
     w' = w + f_bar * d
     ```

  4. Optionally, **track the variance** of `{f_i}` as an indicator of how much the batch agrees on that direction.

### 4. Recompute Error Signals `dy`

- After updating all neurons’ weights in the layer, recompute the outputs `y` and error signals `dy` for each sample.

### 5. Backpropagate through Activation

- Let `x` be the input activations to this layer (the layer’s input or the previous layer’s output).
- We want to find how `x` should change along `d` to reduce the new `dy`.
- Since `d` is the chosen direction, for each sample:

     ```
     f = dy / <x, d>
     ```

  (potentially averaged over **neurons** if they share the same `x`).

- Update the input activation:

     ```
     x' = x + f * d
     ```

### 6. Activation Inverse

- Because the updated `x'` may lie outside the range of standard (e.g., ReLU) activations, this method:
  - Requires an **activation function** whose range is `R` or is at least **invertible** over the domain of interest.
  - A potential idea is to use `x + relu(x)` or a leaky ReLU variant, which is invertible piecewise.

- Apply the **inverse** of the activation function to obtain the target for the **previous** layer.

---

## Remarks and Considerations

1. **Variance Tracking**  
   - The variance of the factors `{f_i}` is an indicator of alignment among the batch samples. A high variance suggests the chosen direction `d` doesn’t **universally** reduce error.

2. **Layer-by-Layer Updates**  
   - This procedure can be repeated for each layer in **sequence**. After updating one layer, you move on to compute new activation targets for the preceding layer.

3. **Choice of Random Directions**  
   - Using more than one random direction `d` per layer could offer a richer subspace for updates. The trade-off is higher computation.

4. **Potential Instabilities**  
   - Large changes in `w` or `x` may lead to big swings in `dy`. Careful **learning rate** or factor scaling might be required.

5. **Connection to Conjugate Gradients**  
   - Like conjugate gradient methods, you update along a specific direction, compute how well it reduces the residual, and then proceed. However, this network context introduces non-linearities that require iterative refinement.

---

## Conclusion

This algorithm attempts a more **direct** update scheme than standard backpropagation, leveraging random directions and explicitly calculating required changes to weights and activations. While it shows promise in addressing finite-step issues (e.g., undesired cross-terms), practical implementation details—such as activation range, variance control, and stable multi-layer updates—remain important areas for experimentation and refinement.
