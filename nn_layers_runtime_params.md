# Channel-Loaded NN Layers

This document describes `nn_layers_runtime_params.act`, a self-contained ACT
layer library for fixed-point neural-network pipelines.

The file imports only:

```act
import "math/fxp.act";
```

All data, weights, biases, and activations use:

```act
math::fixpoint<8,8>
```

## Blocks

| Block | Purpose |
| --- | --- |
| `Activation` | Applies linear, step, ReLU, or leaky ReLU activation |
| `WindowAssembler` | Converts a row-major image stream into valid sliding windows |
| `WinFork` | Copies one window to multiple filters |
| `MaxPoolCh` | Parameterized `P x P` max-pooling for one channel |
| `MaxPool` | Multi-channel max-pooling wrapper |
| `Flatten` | Converts feature-map streams into parallel flat vector channels |
| `FCLayerSource` | Sends dense-layer weights and biases over channels |
| `FCLayer` | Channel-loaded dense layer with internal activation |
| `ConvLayerSource` | Sends convolution weights and biases over channels |
| `ConvBank` | Channel-loaded convolution filter bank |
| `ConvLayer` | Windowing, fork, convolution bank, and internal activation |

## Activation

```act
Activation<ACT_FN, LEAK> act (in, out);
```

Activation codes:

| `ACT_FN` | Function |
| --- | --- |
| `0` | Linear |
| `1` | Step: `1.0` if input is non-negative, else `0.0` |
| `2` | ReLU |
| `3` | Leaky ReLU using `LEAK` |

`ACT_FN` is a template parameter, so it is selected at elaboration time.

## Convolution Path

`ConvLayer` has runtime-loaded weights and biases:

```act
ConvLayer<IMG_W, IMG_H, K, N_IN_CH, N_OUT, ACT_FN, LEAK> conv (
  pixel_in, w_in, b_in, feat_out
);
```

It instantiates these stages internally:

```text
pixel_in
  -> WindowAssembler
  -> WinFork
  -> ConvBank
  -> Activation
  -> feat_out
```

The convolution is valid convolution:

```text
padding = 0
stride  = 1
```

For an `IMG_W x IMG_H` input and a `K x K` kernel, each output channel emits:

```text
(IMG_W - K + 1) * (IMG_H - K + 1)
```

tokens.

### Conv Weight Order

`ConvLayerSource` sends weights in this order:

```text
filter f, input channel ch, kernel position k
```

Index:

```text
W[f*(N_IN_CH*K*K) + ch*K*K + k]
```

Bias order:

```text
B[f]
```

The layer receives all weights and biases once, then reuses them for every
following input window.

## Pooling

Use `MaxPool` for parameterized max-pooling:

```act
MaxPool<N_CH, FEAT_W, FEAT_H, P> pool (feat_in, pool_out);
```

It performs:

```text
pool size = P x P
stride    = P
```

Output size per channel:

```text
FEAT_W / P by FEAT_H / P
```

`FEAT_W` and `FEAT_H` should be divisible by `P`.

Example 2x2 pooling over two `4x4` feature maps:

```act
MaxPool<2, 4, 4, 2> pool1 (conv_out, pool_out);
```

## Flatten

`Flatten` converts feature-map streams into parallel vector channels:

```act
Flatten<N_CH, FEAT_H, FEAT_W> flat (feat_in, flat_out);
```

Output shape:

```act
chan!(math::fixpoint<8,8>) flat_out[N_CH*FEAT_H*FEAT_W]
```

Output index:

```text
flat_out[(r*FEAT_W + c)*N_CH + ch]
```

So the order is row, column, then channel.

## Fully Connected Path

`FCLayer` has runtime-loaded weights and biases:

```act
FCLayer<N_IN, N_OUT, ACT_FN, LEAK> fc (
  x_in, w_in, b_in, out
);
```

It receives:

```text
N_OUT * N_IN weights
N_OUT biases
```

once at startup. Then it repeatedly receives one full input vector and emits one
activated output vector.

### FC Weight Order

`FCLayerSource` sends weights in this order:

```text
output neuron j, input i
```

Index:

```text
W[j*N_IN + i]
```

Bias order:

```text
B[j]
```

## XOR Example

```act
FCLayerSource<2, 2,
              { 5.0, 5.0,
                5.0, 5.0 },
              { -3.0, -8.0 }> hidden_params (hidden_w, hidden_b);

FCLayer<2, 2, 1, 0.0> hidden_layer (
  input_vec, hidden_w, hidden_b, hidden_out
);
```

`ACT_FN = 1` makes the layer a step-activated dense layer.

## CNN Example

```act
ConvLayer<5, 5, 2, 1, 2, 2, 0.0> conv1 (
  pixel_in, conv_w, conv_b, conv_out
);

MaxPool<2, 4, 4, 2> pool1 (conv_out, pool_out);
Flatten<2, 2, 2> flat1 (pool_out, flat_out);

FCLayer<8, 2, 1, 0.0> fc1 (
  flat_out, fc_w, fc_b, class_out
);
```

This corresponds to:

```text
5x5 image
-> 2x2 valid conv, 2 filters, ReLU
-> 2x2 max pool
-> 8 parallel flat channels
-> step-activated FC classifier
```
