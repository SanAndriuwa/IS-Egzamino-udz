% Network from the drawing: 2 inputs -> 3 linear -> 2 tanh -> 3 sigmoid.
% Weight notation: wJI_L = to neuron J, from neuron I, destination layer L.
% Example: w23_2 connects neuron 3 in layer 1 to neuron 2 in layer 2.
% Bias notation: bJ_L. Neuron output: yJ_L. Weighted sum: vJ_L.
% This script trains ONE example to demonstrate the backpropagation maths.
% It does not demonstrate generalisation to unseen data.
clear;
clc;
close all;

%% Inputs and settings - change these values
x_current = 0.6;              % x(n): current signal sample.
x_previous = 0.2;             % x(n-1): previous signal sample, NOT x(n)-1.
target1 = 1;
target2 = 0;
target3 = 1;
eta = 0.1;                    % Learning rate.
epoch = 10000;                % Set to 1 for the hand-calculated update.

%% Layer 1: 2 inputs -> 3 linear neurons
w11_1 = 0.1; w12_1 = 0.2; b1_1 = 0;
w21_1 = 0.3; w22_1 = 0.4; b2_1 = 0;
w31_1 = 0.5; w32_1 = 0.6; b3_1 = 0;

%% Layer 2: 3 inputs -> 2 tanh neurons
w11_2 = 0.1; w12_2 = 0.2; w13_2 = 0.3; b1_2 = 0;
w21_2 = 0.4; w22_2 = 0.5; w23_2 = 0.6; b2_2 = 0;

%% Layer 3: 2 inputs -> 3 sigmoid output neurons
w11_3 = 0.1; w12_3 = 0.2; b1_3 = 0;
w21_3 = 0.3; w22_3 = 0.4; b2_3 = 0;
w31_3 = 0.5; w32_3 = 0.6; b3_3 = 0;

loss = zeros(1, epoch + 1);

% k counts completed updates: 0 means the original parameters.
% Each pass evaluates the current parameters before the next update.
% There are epoch updates and epoch+1 forward passes.
for k = 0:epoch
    %% 1. Forward propagation - layer 1 (linear)
    v1_1 = w11_1*x_current + w12_1*x_previous + b1_1;
    v2_1 = w21_1*x_current + w22_1*x_previous + b2_1;
    v3_1 = w31_1*x_current + w32_1*x_previous + b3_1;
    y1_1 = v1_1;
    y2_1 = v2_1;
    y3_1 = v3_1;

    %% 2. Forward propagation - layer 2 (tanh)
    v1_2 = w11_2*y1_1 + w12_2*y2_1 + w13_2*y3_1 + b1_2;
    v2_2 = w21_2*y1_1 + w22_2*y2_1 + w23_2*y3_1 + b2_2;
    y1_2 = tanh(v1_2);
    y2_2 = tanh(v2_2);

    %% 3. Forward propagation - layer 3 (sigmoid)
    v1_3 = w11_3*y1_2 + w12_3*y2_2 + b1_3;
    v2_3 = w21_3*y1_2 + w22_3*y2_2 + b2_3;
    v3_3 = w31_3*y1_2 + w32_3*y2_2 + b3_3;
    y1_3 = 1/(1 + exp(-v1_3));
    y2_3 = 1/(1 + exp(-v2_3));
    y3_3 = 1/(1 + exp(-v3_3));

    %% 4. Errors and loss: E = 0.5 * sum((prediction-target)^2)
    e1 = y1_3 - target1;
    e2 = y2_3 - target2;
    e3 = y3_3 - target3;
    E = 0.5*(e1^2 + e2^2 + e3^2);
    loss(k + 1) = E;           % MATLAB indices start at 1.

    if k == 0
        Yinitial = [y1_3; y2_3; y3_3];
        fprintf('Initial outputs: %.8f %.8f %.8f\n', y1_3, y2_3, y3_3);
        fprintf('Initial loss: %.10f\n', E);
    end
    if k == 1
        fprintf('After 1 update: %.8f %.8f %.8f\n', y1_3, y2_3, y3_3);
        fprintf('Loss after 1 update: %.10f\n', E);
    end

    % The last pass only checks the final weights. Do not update again.
    if k == epoch
        break;
    end

    %% 5. Output deltas: delta = dE/dv (gradient sign)
    % Sigmoid derivative: y*(1-y).
    delta1_3 = e1*y1_3*(1-y1_3);
    delta2_3 = e2*y2_3*(1-y2_3);
    delta3_3 = e3*y3_3*(1-y3_3);

    %% 6. Layer 2 deltas: tanh derivative * outgoing weighted deltas
    % For neuron 1, source index stays 1 in w11_3, w21_3, w31_3.
    delta1_2 = (1-y1_2^2)*(w11_3*delta1_3 + w21_3*delta2_3 + w31_3*delta3_3);
    delta2_2 = (1-y2_2^2)*(w12_3*delta1_3 + w22_3*delta2_3 + w32_3*delta3_3);

    %% 7. Layer 1 deltas: linear derivative is 1
    delta1_1 = w11_2*delta1_2 + w21_2*delta2_2;
    delta2_1 = w12_2*delta1_2 + w22_2*delta2_2;
    delta3_1 = w13_2*delta1_2 + w23_2*delta2_2;

    % ALL deltas above use OLD weights. Only now update the parameters.
    % weight_new = weight_old - eta * destination_delta * source_output.
    % bias_new = bias_old - eta * destination_delta.

    %% 8. Update layer 3
    w11_3 = w11_3 - eta*delta1_3*y1_2;
    w12_3 = w12_3 - eta*delta1_3*y2_2;
    b1_3 = b1_3 - eta*delta1_3;
    w21_3 = w21_3 - eta*delta2_3*y1_2;
    w22_3 = w22_3 - eta*delta2_3*y2_2;
    b2_3 = b2_3 - eta*delta2_3;
    w31_3 = w31_3 - eta*delta3_3*y1_2;
    w32_3 = w32_3 - eta*delta3_3*y2_2;
    b3_3 = b3_3 - eta*delta3_3;

    %% 9. Update layer 2
    w11_2 = w11_2 - eta*delta1_2*y1_1;
    w12_2 = w12_2 - eta*delta1_2*y2_1;
    w13_2 = w13_2 - eta*delta1_2*y3_1;
    b1_2 = b1_2 - eta*delta1_2;
    w21_2 = w21_2 - eta*delta2_2*y1_1;
    w22_2 = w22_2 - eta*delta2_2*y2_1;
    w23_2 = w23_2 - eta*delta2_2*y3_1;
    b2_2 = b2_2 - eta*delta2_2;

    %% 10. Update layer 1
    w11_1 = w11_1 - eta*delta1_1*x_current;
    w12_1 = w12_1 - eta*delta1_1*x_previous;
    b1_1 = b1_1 - eta*delta1_1;
    w21_1 = w21_1 - eta*delta2_1*x_current;
    w22_1 = w22_1 - eta*delta2_1*x_previous;
    b2_1 = b2_1 - eta*delta2_1;
    w31_1 = w31_1 - eta*delta3_1*x_current;
    w32_1 = w32_1 - eta*delta3_1*x_previous;
    b3_1 = b3_1 - eta*delta3_1;

    % Next loop pass: forward propagation with the UPDATED parameters.
end

%% Results - no further training here
Yfinal = [y1_3; y2_3; y3_3];
target = [target1; target2; target3];
% Matrices are only for inspecting the results; training above uses scalars.
W1 = [w11_1 w12_1; w21_1 w22_1; w31_1 w32_1];
W2 = [w11_2 w12_2 w13_2; w21_2 w22_2 w23_2];
W3 = [w11_3 w12_3; w21_3 w22_3; w31_3 w32_3];
B1 = [b1_1; b2_1; b3_1];
B2 = [b1_2; b2_2];
B3 = [b1_3; b2_3; b3_3];
fprintf('\nCompleted updates: %d\n', epoch);
fprintf('Final outputs: %.8f %.8f %.8f\n', y1_3, y2_3, y3_3);
fprintf('Final loss: %.10f\n', E);
disp('Columns: target, initial prediction, final prediction');
disp([target Yinitial Yfinal]);

figure;
semilogy(0:epoch, loss, 'LineWidth', 1.5);
xlabel('Completed updates');
ylabel('E = 0.5 * sum((prediction-target)^2)');
title('Training loss for one example');
grid on;

figure;
bar([target Yinitial Yfinal]);
xlabel('Output neuron number');
ylabel('Output value');
legend('Target', 'Initial prediction', 'Final prediction', 'Location', 'best');
title('Three sigmoid outputs');
grid on;
