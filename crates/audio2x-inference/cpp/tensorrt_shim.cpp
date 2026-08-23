#include "tensorrt_shim.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) fprintf(stderr, "[TensorRT] %s\n", message);
    }
};

template <typename T> using TrtPtr = std::unique_ptr<T>;

void error_text(char* dst, size_t cap, const char* message) noexcept {
    if (!dst || cap == 0) return;
    const size_t n = std::min(cap - 1, std::strlen(message));
    std::memcpy(dst, message, n);
    dst[n] = 0;
}
void error_text(char* dst, size_t cap, const std::string& message) noexcept {
    if (!dst || cap == 0) return;
    const size_t n = std::min(cap - 1, message.size());
    std::memcpy(dst, message.data(), n);
    dst[n] = 0;
}
std::vector<char> read_engine(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("failed to open TensorRT engine");
    const auto end = file.tellg();
    if (end <= 0) throw std::runtime_error("TensorRT engine is empty");
    std::vector<char> bytes(static_cast<size_t>(end));
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("failed to read TensorRT engine");
    return bytes;
}
void copy_dims(const nvinfer1::Dims& dims, int64_t* out, int32_t capacity, int32_t* rank) {
    if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS) throw std::runtime_error("invalid TensorRT rank");
    if (rank) *rank = dims.nbDims;
    if (!out && capacity == 0) return;
    if (dims.nbDims && (!out || capacity < dims.nbDims))
        throw std::runtime_error("dimension buffer is too small");
    for (int32_t i = 0; i < dims.nbDims; ++i) out[i] = dims.d[i];
}
} // namespace

struct trt_shim_handle {
    int32_t device_id;
    Logger logger;
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;
};

extern "C" trt_shim_handle* trt_shim_create(const char* path, int32_t device_id,
    char* error, size_t error_capacity) {
    try {
        if (!path || !*path) throw std::runtime_error("engine_path is null or empty");
        const cudaError_t cuda_status = cudaSetDevice(device_id);
        if (cuda_status != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_status));
        auto handle = std::make_unique<trt_shim_handle>();
        handle->device_id = device_id;
        const auto bytes = read_engine(path);
        handle->runtime.reset(nvinfer1::createInferRuntime(handle->logger));
        if (!handle->runtime) throw std::runtime_error("createInferRuntime failed");
        handle->engine.reset(handle->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!handle->engine) throw std::runtime_error("deserializeCudaEngine failed");
        handle->context.reset(handle->engine->createExecutionContext());
        if (!handle->context) throw std::runtime_error("createExecutionContext failed");
        return handle.release();
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown TensorRT error"); }
    return nullptr;
}

extern "C" void trt_shim_destroy(trt_shim_handle* handle) { delete handle; }

extern "C" int32_t trt_shim_tensor_count(const trt_shim_handle* h) {
    return h && h->engine ? h->engine->getNbIOTensors() : -1;
}

extern "C" int32_t trt_shim_tensor_name(const trt_shim_handle* h, int32_t index,
    char* name, size_t capacity, size_t* required, char* error, size_t error_capacity) {
    try {
        if (!h || !h->engine || index < 0 || index >= h->engine->getNbIOTensors()) throw std::runtime_error("invalid tensor index");
        const char* source = h->engine->getIOTensorName(index);
        if (!source) throw std::runtime_error("tensor has no name");
        const size_t n = std::strlen(source);
        if (required) *required = n + 1;
        if (!name && capacity == 0) return 1;
        if (!name || capacity <= n) throw std::runtime_error("name buffer is too small");
        std::memcpy(name, source, n + 1);
        return 1;
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown TensorRT error"); }
    return 0;
}

extern "C" int32_t trt_shim_tensor_info_at(const trt_shim_handle* h, int32_t index,
    trt_shim_tensor_info* info, int64_t* dims, int32_t capacity, int32_t* rank,
    char* error, size_t error_capacity) {
    try {
        if (!h || !h->engine || !info || index < 0 || index >= h->engine->getNbIOTensors()) throw std::runtime_error("invalid argument");
        const char* name = h->engine->getIOTensorName(index);
        info->io_mode = static_cast<int32_t>(h->engine->getTensorIOMode(name));
        info->data_type = static_cast<int32_t>(h->engine->getTensorDataType(name));
        info->location = static_cast<int32_t>(h->engine->getTensorLocation(name));
        copy_dims(h->engine->getTensorShape(name), dims, capacity, rank);
        info->rank = rank ? *rank : h->engine->getTensorShape(name).nbDims;
        return 1;
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown TensorRT error"); }
    return 0;
}

extern "C" int32_t trt_shim_tensor_profile_dims(const trt_shim_handle* h, int32_t index,
    int32_t profile, int32_t selector, int64_t* dims, int32_t capacity, int32_t* rank,
    char* error, size_t error_capacity) {
    try {
        if (!h || !h->engine || index < 0 || index >= h->engine->getNbIOTensors()) throw std::runtime_error("invalid tensor index");
        if (profile < 0 || profile >= h->engine->getNbOptimizationProfiles() || selector < 0 || selector > 2) throw std::runtime_error("invalid profile or selector");
        const char* name = h->engine->getIOTensorName(index);
        copy_dims(h->engine->getProfileShape(name, profile, static_cast<nvinfer1::OptProfileSelector>(selector)), dims, capacity, rank);
        return 1;
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown TensorRT error"); }
    return 0;
}

extern "C" int32_t trt_shim_profile_count(const trt_shim_handle* h) {
    return h && h->engine ? h->engine->getNbOptimizationProfiles() : -1;
}

extern "C" int32_t trt_shim_set_profile(trt_shim_handle* h, int32_t profile, void* stream, char* error, size_t cap) {
    try {
        if (!h || !h->context || profile < 0 || profile >= h->engine->getNbOptimizationProfiles()) throw std::runtime_error("invalid profile");
        if (!h->context->setOptimizationProfileAsync(profile, static_cast<cudaStream_t>(stream))) throw std::runtime_error("setOptimizationProfileAsync failed");
        return 1;
    } catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}

extern "C" int32_t trt_shim_set_input_shape(trt_shim_handle* h, const char* name, const int64_t* values, int32_t rank, char* error, size_t cap) {
    try {
        if (!h || !h->context || !name || !values || rank < 0 || rank > nvinfer1::Dims::MAX_DIMS) throw std::runtime_error("invalid shape argument");
        nvinfer1::Dims dims{}; dims.nbDims = rank;
        for (int32_t i = 0; i < rank; ++i) dims.d[i] = values[i];
        if (!h->context->setInputShape(name, dims)) throw std::runtime_error("setInputShape failed");
        return 1;
    } catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}

extern "C" int32_t trt_shim_set_tensor_address(trt_shim_handle* h, const char* name, void* address, char* error, size_t cap) {
    try { if (!h || !h->context || !name || !address || !h->context->setTensorAddress(name, address)) throw std::runtime_error("setTensorAddress failed"); return 1; }
    catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}

extern "C" int32_t trt_shim_enqueue(trt_shim_handle* h, void* stream, char* error, size_t cap) {
    try { if (!h || !h->context || !h->context->enqueueV3(static_cast<cudaStream_t>(stream))) throw std::runtime_error("enqueueV3 failed"); return 1; }
    catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}
