#ifndef AUDIO2X_TENSORRT_SHIM_H
#define AUDIO2X_TENSORRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trt_shim_handle trt_shim_handle;

typedef struct trt_shim_tensor_info {
    int32_t io_mode;
    int32_t data_type;
    int32_t location;
    int32_t rank;
} trt_shim_tensor_info;

typedef struct trt_shim_environment_info {
    int32_t tensorrt_major, tensorrt_minor, tensorrt_patch;
    int32_t cuda_runtime_version, cuda_driver_version;
    int32_t compute_capability_major, compute_capability_minor;
} trt_shim_environment_info;

typedef enum trt_shim_profile_selector {
    TRT_SHIM_PROFILE_MIN = 0,
    TRT_SHIM_PROFILE_OPT = 1,
    TRT_SHIM_PROFILE_MAX = 2,
} trt_shim_profile_selector;

/* Logger entries are retained by the session until destruction. */
int32_t trt_shim_log_count(const trt_shim_handle* handle);
int32_t trt_shim_log_message(const trt_shim_handle* handle, int32_t index,
    char* message, size_t message_capacity, size_t* required_size,
    int32_t* severity, char* error, size_t error_capacity);

trt_shim_handle* trt_shim_create(const char* engine_path, int32_t device_id,
    char* error, size_t error_capacity);
int32_t trt_shim_environment(const trt_shim_handle* handle,
    trt_shim_environment_info* info, char* gpu_name, size_t gpu_name_capacity,
    size_t* required_size, char* error, size_t error_capacity);
void trt_shim_destroy(trt_shim_handle* handle);

int32_t trt_shim_tensor_count(const trt_shim_handle* handle);
int32_t trt_shim_tensor_name(const trt_shim_handle* handle, int32_t index,
    char* name, size_t name_capacity, size_t* required_size,
    char* error, size_t error_capacity);
int32_t trt_shim_tensor_info_at(const trt_shim_handle* handle, int32_t index,
    trt_shim_tensor_info* info, int64_t* dimensions, int32_t dimension_capacity,
    int32_t* rank, char* error, size_t error_capacity);
int32_t trt_shim_tensor_profile_dims(const trt_shim_handle* handle, int32_t index,
    int32_t profile_index, int32_t selector, int64_t* dimensions,
    int32_t dimension_capacity, int32_t* rank, char* error, size_t error_capacity);
int32_t trt_shim_profile_count(const trt_shim_handle* handle);

int32_t trt_shim_set_profile(trt_shim_handle* handle, int32_t profile_index,
    void* cuda_stream, char* error, size_t error_capacity);
int32_t trt_shim_set_input_shape(trt_shim_handle* handle, const char* name,
    const int64_t* dimensions, int32_t rank, char* error, size_t error_capacity);
int32_t trt_shim_infer_shapes(trt_shim_handle* handle,
    char* error, size_t error_capacity);
int32_t trt_shim_context_tensor_dims(const trt_shim_handle* handle, int32_t index,
    int64_t* dimensions, int32_t dimension_capacity, int32_t* rank,
    char* error, size_t error_capacity);
int32_t trt_shim_set_tensor_address(trt_shim_handle* handle, const char* name,
    void* address, char* error, size_t error_capacity);
int32_t trt_shim_enqueue(trt_shim_handle* handle, void* cuda_stream,
    char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
