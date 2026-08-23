#include "tensorrt_shim.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
class Logger final : public nvinfer1::ILogger {
public:
    struct Entry { int32_t severity; std::string message; };

    void log(Severity severity, const char* message) noexcept override {
        try {
            const char* text = message ? message : "(null TensorRT message)";
            {
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.push_back(Entry{static_cast<int32_t>(severity), text});
            }
            if (severity <= Severity::kWARNING) fprintf(stderr, "[TensorRT] %s\n", text);
        } catch (...) {
            // ILogger::log is noexcept; logging must never abort TensorRT.
        }
    }

    int32_t count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int32_t>(entries_.size());
    }
    Entry entry(int32_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < 0 || static_cast<size_t>(index) >= entries_.size())
            throw std::runtime_error("invalid log index");
        return entries_[static_cast<size_t>(index)];
    }

    void record_error(nvinfer1::ErrorCode code, const char* message) noexcept {
        log(Severity::kERROR, (std::string("ErrorRecorder(")
            + std::to_string(static_cast<int32_t>(code)) + "): "
            + (message ? message : "(null)")).c_str());
    }

private:
    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

class ErrorRecorder final : public nvinfer1::IErrorRecorder {
public:
    void set_logger(Logger* logger) noexcept { logger_ = logger; }
    int32_t getNbErrors() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex_); return static_cast<int32_t>(errors_.size());
    }
    nvinfer1::ErrorCode getErrorCode(int32_t index) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        return index >= 0 && static_cast<size_t>(index) < errors_.size()
            ? errors_[static_cast<size_t>(index)].first : nvinfer1::ErrorCode::kUNSPECIFIED_ERROR;
    }
    ErrorDesc getErrorDesc(int32_t index) const noexcept override {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            thread_local std::string description;
            description = index >= 0 && static_cast<size_t>(index) < errors_.size()
                ? errors_[static_cast<size_t>(index)].second : std::string{};
            return description.c_str();
        } catch (...) { return "failed to retrieve TensorRT error description"; }
    }
    bool hasOverflowed() const noexcept override { return overflowed_.load(); }
    void clear() noexcept override {
        try { std::lock_guard<std::mutex> lock(mutex_); errors_.clear(); overflowed_.store(false); } catch (...) {}
    }
    bool reportError(nvinfer1::ErrorCode code, ErrorDesc description) noexcept override {
        try {
            { std::lock_guard<std::mutex> lock(mutex_);
              if (errors_.size() < 64) errors_.emplace_back(code, description ? description : "(null)");
              else overflowed_.store(true); }
            if (logger_) logger_->record_error(code, description);
        } catch (...) { overflowed_.store(true); }
        return false;
    }
    RefCount incRefCount() noexcept override { return ++references_; }
    RefCount decRefCount() noexcept override { return --references_; }
private:
    Logger* logger_{nullptr};
    mutable std::mutex mutex_;
    std::vector<std::pair<nvinfer1::ErrorCode, std::string>> errors_;
    std::atomic<bool> overflowed_{false};
    std::atomic<RefCount> references_{0};
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
std::string cuda_suffix() {
    const cudaError_t status = cudaPeekAtLastError();
    if (status == cudaSuccess) return {};
    return std::string("; CUDA: ") + cudaGetErrorString(status);
}
std::string native_error(const char* operation) {
    return std::string(operation) + " failed" + cuda_suffix();
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

void ensure_device(const trt_shim_handle* handle);
} // namespace

struct trt_shim_handle {
    int32_t device_id;
    Logger logger;
    ErrorRecorder error_recorder;
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;
};

namespace {
void ensure_device(const trt_shim_handle* handle) {
    if (!handle) throw std::runtime_error("TensorRT handle is null");
    int32_t current = -1;
    const cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(status));
    if (current != handle->device_id)
        throw std::runtime_error("current CUDA device does not match TensorRT session device");
}
} // namespace

extern "C" trt_shim_handle* trt_shim_create(const char* path, int32_t device_id,
    char* error, size_t error_capacity) {
    try {
        if (!path || !*path) throw std::runtime_error("engine_path is null or empty");
        const cudaError_t cuda_status = cudaSetDevice(device_id);
        if (cuda_status != cudaSuccess)
            throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cuda_status));
        auto handle = std::make_unique<trt_shim_handle>();
        handle->device_id = device_id;
        handle->error_recorder.set_logger(&handle->logger);
        const auto bytes = read_engine(path);
        handle->runtime.reset(nvinfer1::createInferRuntime(handle->logger));
        if (!handle->runtime) throw std::runtime_error(native_error("createInferRuntime"));
        handle->runtime->setErrorRecorder(&handle->error_recorder);
        handle->engine.reset(handle->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!handle->engine) throw std::runtime_error(native_error("deserializeCudaEngine"));
        handle->engine->setErrorRecorder(&handle->error_recorder);
        handle->context.reset(handle->engine->createExecutionContext());
        if (!handle->context) throw std::runtime_error(native_error("createExecutionContext"));
        handle->context->setErrorRecorder(&handle->error_recorder);
        return handle.release();
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown TensorRT error"); }
    return nullptr;
}

extern "C" void trt_shim_destroy(trt_shim_handle* handle) { delete handle; }

extern "C" int32_t trt_shim_log_count(const trt_shim_handle* h) {
    return h ? h->logger.count() : -1;
}

extern "C" int32_t trt_shim_environment(const trt_shim_handle* h,
    trt_shim_environment_info* info, char* gpu_name, size_t capacity,
    size_t* required, char* error, size_t error_capacity) {
    try {
        ensure_device(h);
        if (!info) throw std::runtime_error("environment info pointer is null");
        cudaDeviceProp properties{};
        int32_t runtime_version = 0, driver_version = 0;
        if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess)
            throw std::runtime_error(native_error("cudaRuntimeGetVersion"));
        if (cudaDriverGetVersion(&driver_version) != cudaSuccess)
            throw std::runtime_error(native_error("cudaDriverGetVersion"));
        if (cudaGetDeviceProperties(&properties, h->device_id) != cudaSuccess)
            throw std::runtime_error(native_error("cudaGetDeviceProperties"));
        *info = {NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
            runtime_version, driver_version, properties.major, properties.minor};
        const size_t n = std::strlen(properties.name);
        if (required) *required = n + 1;
        if (!gpu_name && capacity == 0) return 1;
        if (!gpu_name || capacity <= n) throw std::runtime_error("GPU name buffer is too small");
        std::memcpy(gpu_name, properties.name, n + 1);
        return 1;
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown environment error"); }
    return 0;
}

extern "C" int32_t trt_shim_log_message(const trt_shim_handle* h, int32_t index,
    char* message, size_t capacity, size_t* required, int32_t* severity,
    char* error, size_t error_capacity) {
    try {
        if (!h) throw std::runtime_error("TensorRT handle is null");
        const auto entry = h->logger.entry(index);
        const size_t n = entry.message.size();
        if (required) *required = n + 1;
        if (severity) *severity = entry.severity;
        if (!message && capacity == 0) return 1;
        if (!message || capacity <= n) throw std::runtime_error("log message buffer is too small");
        std::memcpy(message, entry.message.data(), n);
        message[n] = 0;
        return 1;
    } catch (const std::exception& e) { error_text(error, error_capacity, e.what()); }
      catch (...) { error_text(error, error_capacity, "unknown TensorRT log error"); }
    return 0;
}

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
        ensure_device(h);
        if (!h || !h->context || profile < 0 || profile >= h->engine->getNbOptimizationProfiles()) throw std::runtime_error("invalid profile");
        if (!h->context->setOptimizationProfileAsync(profile, static_cast<cudaStream_t>(stream))) throw std::runtime_error(native_error("setOptimizationProfileAsync"));
        return 1;
    } catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}

extern "C" int32_t trt_shim_set_input_shape(trt_shim_handle* h, const char* name, const int64_t* values, int32_t rank, char* error, size_t cap) {
    try {
        ensure_device(h);
        if (!h || !h->context || !name || !values || rank < 0 || rank > nvinfer1::Dims::MAX_DIMS) throw std::runtime_error("invalid shape argument");
        nvinfer1::Dims dims{}; dims.nbDims = rank;
        for (int32_t i = 0; i < rank; ++i) dims.d[i] = values[i];
        if (!h->context->setInputShape(name, dims)) throw std::runtime_error(native_error("setInputShape"));
        return 1;
    } catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}

extern "C" int32_t trt_shim_infer_shapes(trt_shim_handle* h, char* error, size_t cap) {
    try {
        ensure_device(h);
        if (!h->context) throw std::runtime_error("execution context is null");
        const int32_t missing = h->context->inferShapes(0, nullptr);
        if (missing < 0) throw std::runtime_error(native_error("inferShapes"));
        if (missing != 0) throw std::runtime_error("not all input dimensions are specified");
        return 1;
    } catch (const std::exception& e) { error_text(error, cap, e.what()); }
      catch (...) { error_text(error, cap, "unknown TensorRT error"); }
    return 0;
}

extern "C" int32_t trt_shim_context_tensor_dims(const trt_shim_handle* h, int32_t index,
    int64_t* dims, int32_t capacity, int32_t* rank, char* error, size_t cap) {
    try {
        ensure_device(h);
        if (!h->context || index < 0 || index >= h->engine->getNbIOTensors())
            throw std::runtime_error("invalid tensor index");
        const char* name = h->engine->getIOTensorName(index);
        copy_dims(h->context->getTensorShape(name), dims, capacity, rank);
        return 1;
    } catch (const std::exception& e) { error_text(error, cap, e.what()); }
      catch (...) { error_text(error, cap, "unknown TensorRT error"); }
    return 0;
}

extern "C" int32_t trt_shim_set_tensor_address(trt_shim_handle* h, const char* name, void* address, char* error, size_t cap) {
    try { ensure_device(h); if (!h->context || !name || !address || !h->context->setTensorAddress(name, address)) throw std::runtime_error(native_error("setTensorAddress")); return 1; }
    catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}

extern "C" int32_t trt_shim_enqueue(trt_shim_handle* h, void* stream, char* error, size_t cap) {
    try { ensure_device(h); if (!h->context || !h->context->enqueueV3(static_cast<cudaStream_t>(stream))) throw std::runtime_error(native_error("enqueueV3")); return 1; }
    catch (const std::exception& e) { error_text(error, cap, e.what()); } catch (...) { error_text(error, cap, "unknown TensorRT error"); } return 0;
}
