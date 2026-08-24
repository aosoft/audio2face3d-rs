#include <math.h>

namespace {

__device__ unsigned int packed_track_index(const unsigned long long* active, unsigned int track)
{
    const unsigned int word = track / 64U;
    unsigned int packed = 0;
    for (unsigned int i = 0; i < word; ++i) packed += __popcll(active[i]);
    const unsigned int remaining = track % 64U;
    const unsigned long long before = active[word] & ((1ULL << remaining) - 1ULL);
    return packed + __popcll(before);
}

__device__ void process_track(
    float* output, unsigned long long output_stride,
    const float* input, unsigned long long input_stride,
    const int* correspondence, const float* parameters,
    unsigned long long parameter_stride, const float* preferred,
    float* state, unsigned long long state_stride,
    const unsigned long long* active, unsigned int input_length,
    unsigned int output_length, unsigned int track)
{
    if ((active[track / 64U] & (1ULL << (track % 64U))) == 0) return;
    const unsigned int packed = packed_track_index(active, track);
    const float* source = input + packed * input_stride;
    float* target = output + track * output_stride;
    const float* params = parameters + track * parameter_stride;
    const float* selected_preferred = preferred + track * output_length;
    float* track_state = state + track * state_stride;
    float* inference = track_state + 1;
    float* working = inference + input_length;
    float* previous = working + output_length;
    float* previous_blended = previous + output_length;
    const bool first_frame = track_state[0] != 0.0f;

    float maximum = -INFINITY;
    for (unsigned int i = 0; i < input_length; ++i) {
        inference[i] = source[i] * params[0];
        maximum = fmaxf(maximum, inference[i]);
    }
    float sum = 0.0f;
    for (unsigned int i = 0; i < input_length; ++i) {
        inference[i] = expf(inference[i] - maximum);
        sum += inference[i];
    }
    for (unsigned int i = 0; i < input_length; ++i) {
        inference[i] /= sum;
        if (correspondence[i] == -1) inference[i] = 0.0f;
    }

    const unsigned int max_emotions = static_cast<unsigned int>(params[1]);
    if (max_emotions < input_length) {
        for (unsigned int removed = 0; removed < input_length - max_emotions; ++removed) {
            unsigned int smallest = input_length;
            float value = INFINITY;
            for (unsigned int i = 0; i < input_length; ++i) {
                if (inference[i] < value) {
                    value = inference[i];
                    smallest = i;
                }
            }
            if (smallest < input_length) inference[smallest] = INFINITY;
        }
        for (unsigned int i = 0; i < input_length; ++i) {
            if (isinf(inference[i])) inference[i] = 0.0f;
        }
    }

    if (first_frame) {
        for (unsigned int i = 0; i < output_length; ++i) working[i] = params[8 + i];
    }
    for (unsigned int i = 0; i < input_length; ++i) {
        const int mapped = correspondence[i];
        if (mapped >= 0) working[mapped] = inference[i];
    }
    const float blend = params[2];
    for (unsigned int i = 0; i < output_length; ++i) {
        const float source_value = first_frame ? params[8 + i] : previous[i];
        working[i] = (1.0f - blend) * working[i] + blend * source_value;
        previous[i] = working[i];
    }
    if (params[3] != 0.0f) {
        const float preferred_strength = params[4];
        for (unsigned int i = 0; i < output_length; ++i) {
            working[i] = (1.0f - preferred_strength) * working[i]
                + preferred_strength * selected_preferred[i];
        }
    }
    if (!first_frame) {
        const float transition = fmaxf(params[5], 1.0e-3f);
        const float weight = fminf(params[6] / transition, 1.0f);
        for (unsigned int i = 0; i < output_length; ++i) {
            working[i] = weight * working[i] + (1.0f - weight) * previous_blended[i];
        }
    }
    for (unsigned int i = 0; i < output_length; ++i) {
        previous_blended[i] = working[i];
        target[i] = working[i] * params[7];
    }
    track_state[0] = 0.0f;
}

} // namespace

extern "C" __global__ void emotion_postprocess_set(
    unsigned long long* destination, unsigned int word, unsigned long long value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) destination[word] = value;
}

extern "C" __global__ void emotion_postprocess_reset(
    float* state, unsigned long long state_stride, unsigned int track_count)
{
    const unsigned int track = blockIdx.x * blockDim.x + threadIdx.x;
    if (track < track_count) state[track * state_stride] = 1.0f;
}

extern "C" __global__ void emotion_postprocess_generic(
    float* output, unsigned long long output_stride,
    const float* input, unsigned long long input_stride,
    const int* correspondence, const float* parameters,
    unsigned long long parameter_stride, const float* preferred,
    float* state, unsigned long long state_stride,
    const unsigned long long* active, unsigned int input_length,
    unsigned int output_length, unsigned int track_count)
{
    const unsigned int track = blockIdx.x * blockDim.x + threadIdx.x;
    if (track < track_count) process_track(output, output_stride, input, input_stride,
        correspondence, parameters, parameter_stride, preferred, state, state_stride,
        active, input_length, output_length, track);
}

extern "C" __global__ void emotion_postprocess_local(
    float* output, unsigned long long output_stride,
    const float* input, unsigned long long input_stride,
    const int* correspondence, const float* parameters,
    unsigned long long parameter_stride, const float* preferred,
    float* state, unsigned long long state_stride,
    const unsigned long long* active, unsigned int input_length,
    unsigned int output_length, unsigned int track_count)
{
    const unsigned int track = blockIdx.x;
    if (threadIdx.x == 0 && track < track_count) process_track(output, output_stride, input,
        input_stride, correspondence, parameters, parameter_stride, preferred, state,
        state_stride, active, input_length, output_length, track);
}
