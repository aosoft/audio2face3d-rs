// Device-resident lower-teeth reconstruction for the regression pipeline.
// One block processes one track; only thread zero is used because jaw meshes are
// small and the eigensolve is inherently serial.

#include <math.h>
#include <stdint.h>

__device__ __forceinline__ bool track_is_set(
    const unsigned long long* tracks, uint32_t track) {
    return tracks == nullptr || ((tracks[track / 64] >> (track % 64)) & 1ULL) != 0;
}

namespace {

__device__ void largest_eigenvector(double matrix[4][4], double result[4]) {
    double vectors[4][4] = {};
    for (int index = 0; index < 4; ++index) {
        vectors[index][index] = 1.0;
    }

    for (int sweep = 0; sweep < 32; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 4; ++q) {
                if (fabs(matrix[p][q]) <= 2.2204460492503131e-16) {
                    continue;
                }
                const double angle =
                    0.5 * atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
                double sine;
                double cosine;
                sincos(angle, &sine, &cosine);

                for (int row = 0; row < 4; ++row) {
                    const double a = matrix[row][p];
                    const double b = matrix[row][q];
                    matrix[row][p] = cosine * a - sine * b;
                    matrix[row][q] = sine * a + cosine * b;
                }
                for (int column = 0; column < 4; ++column) {
                    const double a = matrix[p][column];
                    const double b = matrix[q][column];
                    matrix[p][column] = cosine * a - sine * b;
                    matrix[q][column] = sine * a + cosine * b;
                }
                for (int row = 0; row < 4; ++row) {
                    const double a = vectors[row][p];
                    const double b = vectors[row][q];
                    vectors[row][p] = cosine * a - sine * b;
                    vectors[row][q] = sine * a + cosine * b;
                }
            }
        }
    }

    int largest = 0;
    for (int index = 1; index < 4; ++index) {
        if (matrix[index][index] > matrix[largest][largest]) {
            largest = index;
        }
    }
    for (int row = 0; row < 4; ++row) {
        result[row] = vectors[row][largest];
    }
}

}  // namespace

// neutral_pose contains point_count interleaved xyz positions. jaw_deltas uses
// track-major layout [track_count][point_count][xyz]. transforms uses
// track-major layout [track_count][16], with each matrix stored column-major.
extern "C" __global__ void audio2face_regression_jaw(
    const float* neutral_pose,
    const float* jaw_deltas,
    uint32_t point_count,
    uint32_t track_count,
    const float* params,
    uint32_t params_stride,
    const unsigned long long* active_tracks,
    float* transforms) {
    const uint32_t track = blockIdx.x;
    if (track >= track_count || threadIdx.x != 0 || point_count == 0
        || !track_is_set(active_tracks, track)) {
        return;
    }

    const float* track_params = params + static_cast<size_t>(track) * params_stride;
    const float strength = track_params[0];
    const float height_offset = track_params[1];
    const float depth_offset = track_params[2];

    const float* deltas = jaw_deltas + static_cast<size_t>(track) * point_count * 3;
    double source_mean[3] = {};
    double target_mean[3] = {};
    for (uint32_t point = 0; point < point_count; ++point) {
        const size_t offset = static_cast<size_t>(point) * 3;
        for (int component = 0; component < 3; ++component) {
            const double source = neutral_pose[offset + component];
            double target = source + static_cast<double>(deltas[offset + component]) * strength;
            if (component == 1) {
                target += height_offset;
            } else if (component == 2) {
                target += depth_offset;
            }
            source_mean[component] += source;
            target_mean[component] += target;
        }
    }
    for (int component = 0; component < 3; ++component) {
        source_mean[component] /= point_count;
        target_mean[component] /= point_count;
    }

    double covariance[3][3] = {};
    for (uint32_t point = 0; point < point_count; ++point) {
        const size_t offset = static_cast<size_t>(point) * 3;
        double source[3];
        double target[3];
        for (int component = 0; component < 3; ++component) {
            source[component] = neutral_pose[offset + component] - source_mean[component];
            target[component] = neutral_pose[offset + component]
                + static_cast<double>(deltas[offset + component]) * strength
                - target_mean[component];
            if (component == 1) {
                target[component] += height_offset;
            } else if (component == 2) {
                target[component] += depth_offset;
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance[row][column] += source[row] * target[column];
            }
        }
    }

    const double trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
    double horn[4][4] = {
        {trace, covariance[1][2] - covariance[2][1], covariance[2][0] - covariance[0][2], covariance[0][1] - covariance[1][0]},
        {covariance[1][2] - covariance[2][1], covariance[0][0] - covariance[1][1] - covariance[2][2], covariance[0][1] + covariance[1][0], covariance[0][2] + covariance[2][0]},
        {covariance[2][0] - covariance[0][2], covariance[0][1] + covariance[1][0], -covariance[0][0] + covariance[1][1] - covariance[2][2], covariance[1][2] + covariance[2][1]},
        {covariance[0][1] - covariance[1][0], covariance[0][2] + covariance[2][0], covariance[1][2] + covariance[2][1], -covariance[0][0] - covariance[1][1] + covariance[2][2]},
    };
    double quaternion[4];
    largest_eigenvector(horn, quaternion);
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    const double rotation[3][3] = {
        {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)},
        {2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)},
        {2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)},
    };

    float* output = transforms + static_cast<size_t>(track) * 16;
    for (int index = 0; index < 16; ++index) {
        output[index] = 0.0f;
    }
    output[15] = 1.0f;
    for (int row = 0; row < 3; ++row) {
        double translation = target_mean[row];
        for (int column = 0; column < 3; ++column) {
            output[column * 4 + row] = static_cast<float>(rotation[row][column]);
            translation -= rotation[row][column] * source_mean[column];
        }
        output[12 + row] = static_cast<float>(translation);
    }
}
