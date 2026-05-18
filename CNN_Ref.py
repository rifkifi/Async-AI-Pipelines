import numpy as np

# =========================================================
# Input Image (same as simple_cnn.act testbench)
# =========================================================

img = np.array([
    [1,1,0,0,0],
    [1,1,0,0,0],
    [1,1,0,0,0],
    [1,1,0,0,0],
    [1,1,0,0,0]
], dtype=float)

print("\nINPUT IMAGE")
print(img)

# =========================================================
# Convolution Filters (same as ACT)
# =========================================================

f0 = np.array([
    [0.25, 0.25],
    [0.25, 0.25]
])

f1 = np.array([
    [1, -1],
    [1, -1]
])

# =========================================================
# 2D Convolution
# =========================================================

def conv2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape

    out = np.zeros((h-kh+1, w-kw+1))

    for i in range(h-kh+1):
        for j in range(w-kw+1):
            region = img[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * kernel)

    return out

# =========================================================
# Convolution
# =========================================================

c0 = conv2d(img, f0)
c1 = conv2d(img, f1)

print("\nCONV FILTER 0")
print(c0)

print("\nCONV FILTER 1")
print(c1)

# =========================================================
# ReLU
# =========================================================

c0 = np.maximum(c0, 0)
c1 = np.maximum(c1, 0)

print("\nRELU FILTER 0")
print(c0)

print("\nRELU FILTER 1")
print(c1)

# =========================================================
# MaxPool 2x2
# =========================================================

def maxpool2x2(x):
    h, w = x.shape

    out = np.zeros((h//2, w//2))

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            out[i//2, j//2] = np.max(x[i:i+2, j:j+2])

    return out

p0 = maxpool2x2(c0)
p1 = maxpool2x2(c1)

print("\nMAXPOOL FILTER 0")
print(p0)

print("\nMAXPOOL FILTER 1")
print(p1)

# =========================================================
# Flatten
# ACT ordering:
# flat_out[(r*FEAT_W + c)*N_CH + ch]
# =========================================================

flat = []

for r in range(2):
    for c in range(2):
        flat.append(p0[r,c])
        flat.append(p1[r,c])

flat = np.array(flat)

print("\nFLATTEN OUTPUT")
print(flat)

# Expected:
# [1,2,0,0,1,2,0,0]

# =========================================================
# Fully Connected Layer
# Same as simple_cnn.act
# =========================================================

W = np.array([
    [1,0,1,0,1,0,1,0],
    [0,1,0,1,0,1,0,1]
])

B = np.array([
    -0.5,
    -4.5
])

# =========================================================
# FC Compute
# =========================================================

y_raw = W @ flat + B

print("\nFC RAW OUTPUT")
print(y_raw)

# =========================================================
# Step Activation
# =========================================================

y = (y_raw >= 0).astype(int)

print("\nFINAL CLASS OUTPUT")
print(y)


