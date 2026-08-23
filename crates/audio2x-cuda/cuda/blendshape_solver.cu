extern "C" __global__ void blendshape_gather_subtract(
    float* result, const float* target, const float* neutral,
    const unsigned int* indices, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) result[i] = target[indices[i]] - neutral[i];
}

extern "C" __global__ void blendshape_fill(float* values, float value, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = value;
}

extern "C" __global__ void blendshape_copy(float* result, const float* source, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) result[i] = source[i];
}

extern "C" __global__ void blendshape_clip(
    float* values, const float* lower, const float* upper, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = fminf(fmaxf(values[i], lower[i]), upper[i]);
}

extern "C" __global__ void blendshape_admm_update(
    float* u_out, float* z_out, const float* u, const float* z,
    const float* weights, const float* atb, const float* inverse,
    const float* lower, const float* upper, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = 0.0f;
    for (unsigned int j = 0; j < count; ++j) {
        float rhs = weights[j] * weights[j] * (z[j] - u[j]) + atb[j];
        x += inverse[j + count * i] * rhs;
    }
    z_out[i] = fminf(fmaxf(u[i] + x, lower[i]), upper[i]);
    u_out[i] = u[i] + 1.9f * (x - z_out[i]);
}

extern "C" __global__ void blendshape_cancel_upper(
    float* upper, const float* weights, const unsigned int* first,
    const unsigned int* second, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned int a = first[i];
        unsigned int b = second[i];
        upper[weights[a] >= weights[b] ? b : a] = 1.0e-10f;
    }
}

extern "C" __global__ void blendshape_unmap(
    float* full, const float* active, const unsigned int* indices, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) full[indices[i]] = active[i];
}

extern "C" __global__ void blendshape_apply(
    float* weights, const float* multipliers, const float* offsets, unsigned int count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) weights[i] = weights[i] * multipliers[i] + offsets[i];
}
