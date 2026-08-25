// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Device-resident, multi-track regression post-processing. The equations and
// parameter layout follow Audio2Face-3D-SDK multitrack_animator_cuda.cu.

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

namespace {

__device__ __forceinline__ bool track_is_set(
    const unsigned long long* tracks, size_t track) {
  return tracks == nullptr || ((tracks[track / 64] >> (track % 64)) & 1ULL) != 0;
}

}  // namespace

// params per track: skin strength, eyelid-open offset, blink offset,
// blink strength, lip-open offset, lower alpha, upper alpha, lower strength,
// upper strength. interp stores lower1/lower2/upper1/upper2 per pose element.
extern "C" __global__ void audio2face3d_skin_postprocess(
    float* results, size_t results_offset, size_t results_stride,
    const float* input_deltas, size_t input_offset, size_t input_stride,
    const float* animator_data, size_t animator_data_stride,
    const float* face_mask_lower,
    float* interp, size_t interp_stride,
    const float* params, size_t params_stride,
    const unsigned long long* active_tracks,
    unsigned long long* initialized_tracks,
    size_t pose_size, size_t track_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pose_size * track_count) return;
  const size_t track = index / pose_size;
  if (!track_is_set(active_tracks, track)) return;
  const size_t element = index % pose_size;
  const bool initialized = track_is_set(initialized_tracks, track);

  const float input = input_deltas[track * input_stride + input_offset + element];
  const float eye = animator_data[element * animator_data_stride + 0];
  const float lip = animator_data[element * animator_data_stride + 1];
  const float neutral = animator_data[element * animator_data_stride + 2];
  const float mask = face_mask_lower[element / 3];
  const float* p = params + track * params_stride;
  const float delta = p[0] * input + eye * (-p[1] + p[2] * p[3]) + lip * p[4];

  float* state = interp + (track * pose_size + element) * interp_stride;
  if (initialized && p[5] > 0.0f) {
    state[0] += (delta - state[0]) * p[5];
    state[1] += (state[0] - state[1]) * p[5];
  } else {
    state[0] = delta;
    state[1] = delta;
  }
  if (initialized && p[6] > 0.0f) {
    state[2] += (delta - state[2]) * p[6];
    state[3] += (state[2] - state[3]) * p[6];
  } else {
    state[2] = delta;
    state[3] = delta;
  }

  results[track * results_stride + results_offset + element] =
      neutral + state[3] * p[8] * (1.0f - mask) + state[1] * p[7] * mask;
}

extern "C" __global__ void audio2face3d_mark_initialized(
    unsigned long long* initialized_tracks,
    const unsigned long long* active_tracks,
    size_t word_count) {
  const size_t word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word < word_count) initialized_tracks[word] |= active_tracks[word];
}

// params per track: strength, height offset, depth offset.
extern "C" __global__ void audio2face3d_tongue_postprocess(
    float* results, size_t results_offset, size_t results_stride,
    const float* input_deltas, size_t input_offset, size_t input_stride,
    const float* neutral_pose,
    const float* params, size_t params_stride,
    const unsigned long long* active_tracks,
    size_t pose_size, size_t track_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pose_size * track_count) return;
  const size_t track = index / pose_size;
  if (!track_is_set(active_tracks, track)) return;
  const size_t element = index % pose_size;
  const float* p = params + track * params_stride;
  float value = neutral_pose[element] +
      input_deltas[track * input_stride + input_offset + element] * p[0];
  if (element % 3 == 1) value += p[1];
  if (element % 3 == 2) value += p[2];
  results[track * results_stride + results_offset + element] = value;
}

// params per track: eyeballs strength, saccade strength, right XY offsets,
// left XY offsets, saccade seed. The six outputs are right XYZ then left XYZ.
extern "C" __global__ void audio2face3d_eyes_postprocess(
    float* output, size_t output_offset, size_t output_stride,
    const float* input, size_t input_offset, size_t input_stride,
    const float* params, size_t params_stride,
    const float* saccade_rotation, size_t saccade_rotation_size,
    float dt, float* live_time,
    const unsigned long long* active_tracks,
    size_t track_count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t track = index / 6;
  if (track >= track_count || !track_is_set(active_tracks, track)) return;
  const size_t element = index % 6;
  const size_t eye = element / 3;
  const size_t axis = element % 3;
  float value = 0.0f;
  if (axis < 2) {
    const float* p = params + track * params_stride;
    const float frame_count = static_cast<float>(saccade_rotation_size / 2);
    float total = fmodf(p[6] + live_time[track], frame_count);
    if (total < 0.0f) total += frame_count;
    const size_t frame = static_cast<size_t>(total);
    const float saccade = saccade_rotation[frame * 2 + axis];
    value = p[2 + eye * 2 + axis]
        + __fmul_rn(input[track * input_stride + input_offset + eye * 2 + axis], p[0])
        + __fmul_rn(p[1], saccade);
    if (element == 0) {
      float next = fmodf(live_time[track] + dt * 30.0f, frame_count);
      if (next < 0.0f) next += frame_count;
      live_time[track] = next;
    }
  }
  output[track * output_stride + output_offset + element] = value;
}
