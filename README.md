# Simple MATLAB backpropagation: 2-3-2-3

One readable script, `testing.m`, in the scalar-variable style of Lab2.
No neural-network toolbox, classes, helper functions or automatic differentiation.

## Run

Open `testing.m` in MATLAB and press **Run**, or enter `testing` in the Command Window with this folder selected.

- Set `epoch = 1` to reproduce one training step from the hand calculations.
- The default `epoch = 10000` repeatedly trains the same example.
- Change `eta` to change the learning rate.
- Change `x_current`, `x_previous`, and `target1` through `target3` at the top.
- Edit the initial weights and biases in the next three sections.

The script prints initial, first-update and final predictions and loss, and draws loss and prediction plots. All scalar weights remain in the MATLAB workspace. `W1`, `W2`, `W3`, `B1`, `B2`, `B3` collect the final values for inspection.

## Network and notation

| Layer | Neurons | Activation | Code outputs |
|---|---:|---|---|
| Input | 2 | None | `x_current`, `x_previous` |
| 1 | 3 | Linear | `y1_1`, `y2_1`, `y3_1` |
| 2 | 2 | tanh | `y1_2`, `y2_2` |
| 3 | 3 | Sigmoid | `y1_3`, `y2_3`, `y3_3` |

`wJI_L` means **to neuron J, from neuron I, destination layer L**.
For example, `w23_2` connects layer-1 neuron 3 to layer-2 neuron 2.
`bJ_L` is that neuron's bias; `vJ_L` is its weighted sum before activation.
The three first-layer outputs correspond to a1,a2,a3 in the hand calculations;
the second-layer outputs correspond to h1,h2.

There are 18 connection weights and 8 biases. Matrix rows are destinations and columns are sources.

`x(n)` is the current sample, `x(n-1)` is the previous sample of the same signal. It is not `x(n)-1`. Here they are 0.6 and 0.2.

## Training equations

For source outputs z_i and layer activation phi:

- `v_j = sum_i(w_ji * z_i) + b_j`
- `y_j = phi(v_j)`
- `E = 0.5 * sum_j((y_j - target_j)^2)`
- Output delta: `(y_j - target_j) * y_j * (1-y_j)`.
- Hidden delta: `phi'(v_j) * sum_k(w_kj_next * delta_k_next)`.
- Linear derivative: `1`. Tanh derivative: `1-y_j^2`.
- Weight update: `w_ji = w_ji - eta * delta_j * z_i`.
- Bias update: `b_j = b_j - eta * delta_j`.

All deltas are calculated before any weight is changed. The next loop pass is forward propagation with the updated parameters.

Unlike some older Lab2 scripts, this script defines error as prediction minus target and delta as dE/dv; therefore updates SUBTRACT the gradient. Do not mix this convention with target-minus-prediction deltas and subtraction. The destination-first weight indexing is used consistently, including the output layer.

The loop uses `k = 0:epoch`: `k` counts completed updates. At `k=0` it evaluates the initial parameters. At `k=epoch` it evaluates the final parameters and stops before another update. Thus there are exactly `epoch` updates and `epoch+1` forward evaluations. With one example, each update is one epoch.

## Expected results

| Stage | Output 1 | Output 2 | Output 3 | Loss |
|---|---:|---:|---:|---:|
| Initial | 0.52455633 | 0.55359095 | 0.58226451 | 0.3535062867 |
| After 1 update | 0.52847812 | 0.55040639 | 0.58653751 | 0.3481156556 |
| After 10000 updates | 0.98678459 | 0.01332007 | 0.98692094 | 0.0002615666 |

Targets are [1; 0; 1]. The sigmoid outputs are independent and need not sum to one.

Validation: scalar expressions extracted directly from the MATLAB script were evaluated independently in Python. The first update matches the hand calculation. All 26 parameter gradients were checked with central finite differences (maximum absolute discrepancy below 9e-11). This is a numerical check, not a claim that MATLAB itself was executed.

## Scope

This is a teaching example that memorises one input/target pair. The drawing does not specify what the three outputs mean, so no real dataset or prediction task is invented. Training on a real time series requires multiple current/previous pairs with three corresponding targets per pair, and separate evaluation data. The displayed loss is training loss, not test accuracy.
