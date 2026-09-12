// SPDX-License-Identifier: MIT
#include <audio2face/audio2face.h>
#include <audio2emotion/audio2emotion.h>
#include <audio2x/audio_accumulator.h>
#include <audio2x/cuda_stream.h>
#include <audio2x/cuda_utils.h>
#include <audio2x/emotion_accumulator.h>
#include <audio2x/executor.h>
#include <audio2x/tensor_float.h>

#include <windows.h>
#include <bcrypt.h>

#include <cstdint>
#include <algorithm>
#include <limits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

template <class T> struct Destroy {
  void operator()(T* value) const noexcept {
    if (value != nullptr) value->Destroy();
  }
};
template <class T> using SdkPtr = std::unique_ptr<T, Destroy<T>>;

static void check(std::error_code error, const char* operation) {
  if (error) throw std::runtime_error(std::string(operation) + ": " + error.message());
}

static std::string hex(const std::vector<unsigned char>& bytes) {
  std::ostringstream result;
  result << std::uppercase << std::hex << std::setfill('0');
  for (const auto value : bytes) result << std::setw(2) << static_cast<unsigned>(value);
  return result.str();
}

static std::string sha256(const void* data, std::size_t size) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_size = 0;
  DWORD copied = 0;
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                        reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &copied, 0) < 0) {
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error("BCrypt SHA-256 initialization failed");
  }
  std::vector<unsigned char> object(object_size);
  std::vector<unsigned char> digest(32);
  const auto cleanup = [&] {
    if (hash != nullptr) BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
  };
  if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0 ||
      (size != 0 && BCryptHashData(hash,
          const_cast<PUCHAR>(static_cast<const unsigned char*>(data)),
          static_cast<ULONG>(size), 0) < 0) ||
      BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0) {
    cleanup();
    throw std::runtime_error("BCrypt SHA-256 calculation failed");
  }
  cleanup();
  return hex(digest);
}

static std::vector<std::uint8_t> read_bytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  input.seekg(0, std::ios::end);
  const auto length = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  input.read(reinterpret_cast<char*>(bytes.data()), length);
  if (!input) throw std::runtime_error("cannot read " + path.string());
  return bytes;
}

static std::string quote(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(character) << std::dec;
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

struct Record {
  std::size_t sequence{};
  std::string layer;
  std::string component;
  std::size_t track{};
  std::size_t frame{};
  std::int64_t timestamp{};
  std::int64_t next_timestamp{};
  std::uint64_t offset{};
  std::uint64_t length{};
  std::string hash;
  std::size_t values{};
};

class ArtifactWriter {
public:
  explicit ArtifactWriter(fs::path root) : root_(std::move(root)) {
    fs::create_directories(root_);
    data_.open(root_ / "values.f32le", std::ios::binary | std::ios::trunc);
    if (!data_) throw std::runtime_error("cannot create values.f32le");
  }

  void push(const char* layer, const char* component, std::size_t track, std::size_t frame,
            std::int64_t timestamp, std::int64_t next_timestamp,
            nva2x::DeviceTensorFloatConstView source) {
    std::vector<float> values(source.Size());
    check(nva2x::CopyDeviceToHost(
              nva2x::HostTensorFloatView(values.data(), values.size()), source),
          "CopyDeviceToHost");
    push_host(layer, component, track, frame, timestamp, next_timestamp,
              values.data(), values.size());
  }

  void push_host(const char* layer, const char* component, std::size_t track,
                 std::size_t frame, std::int64_t timestamp, std::int64_t next_timestamp,
                 const float* source, std::size_t count) {
    const std::scoped_lock lock(mutex_);
    const auto bytes = count * sizeof(float);
    data_.write(reinterpret_cast<const char*>(source), static_cast<std::streamsize>(bytes));
    if (!data_) throw std::runtime_error("cannot write values.f32le");
    records_.push_back(Record{records_.size(), layer, component, track, frame, timestamp,
                              next_timestamp, offset_, bytes,
                              sha256(source, bytes), count});
    offset_ += bytes;
  }

  void finish(const std::string& pipeline, const std::string& execution,
              const std::string& precision,
              std::uint64_t seed, std::size_t tracks, const fs::path& fixture,
              const fs::path& model) {
    if (execution == "blendshape-cpu") {
      // Host jobs on different tracks complete independently. Match the Rust
      // capture's frame/track ordering, retaining arrival order separately.
      std::ofstream arrival(root_ / "callback-order.csv", std::ios::trunc);
      if (!arrival) throw std::runtime_error("cannot create callback order trace");
      arrival << "sequence,track,frame,timestamp\n";
      std::vector<std::size_t> next_frame(tracks, 0);
      std::vector<std::int64_t> last_timestamp(tracks, std::numeric_limits<std::int64_t>::lowest());
      for (const auto& record : records_) {
        if (record.frame != next_frame.at(record.track)++ ||
            record.timestamp <= last_timestamp.at(record.track)) {
          throw std::runtime_error("host callback order regressed within a track");
        }
        last_timestamp[record.track] = record.timestamp;
        arrival << record.sequence << ',' << record.track << ',' << record.frame
                << ',' << record.timestamp << '\n';
      }
      std::stable_sort(records_.begin(), records_.end(), [](const auto& a, const auto& b) {
        return a.frame < b.frame || (a.frame == b.frame && a.track < b.track);
      });
      for (std::size_t index = 0; index < records_.size(); ++index) records_[index].sequence = index;
    }
    data_.flush();
    data_.close();
    const auto data_bytes = read_bytes(root_ / "values.f32le");
    const auto fixture_bytes = read_bytes(fixture);
    const auto model_bytes = read_bytes(model);
    std::ofstream json(root_ / "artifact.json", std::ios::trunc);
    if (!json) throw std::runtime_error("cannot create artifact.json");
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"producer\": {\"implementation\": \"cpp\", \"version\": \"1ca0f02535ed774f5dbcd724a31cd486368dc783\"},\n"
         << "  \"case\": {\"name\": " << quote(pipeline + "-" + execution)
         << ", \"pipeline\": " << quote(pipeline)
         << ", \"execution\": " << quote(execution) << ", \"precision\": " << quote(precision)
         << ", \"seed\": " << seed << ", \"track_count\": " << tracks << "},\n"
         << "  \"fixture\": {\"path\": " << quote(fixture.string())
         << ", \"sha256\": " << quote(sha256(fixture_bytes.data(), fixture_bytes.size())) << "},\n"
         << "  \"model_files\": {\"model-descriptor\": {\"path\": " << quote(model.string())
         << ", \"sha256\": " << quote(sha256(model_bytes.data(), model_bytes.size())) << "}},\n"
         << "  \"environment\": {\"os\": \"windows\", \"arch\": \"x86_64\"},\n"
         << "  \"counters\": {},\n"
         << "  \"records\": [\n";
    for (std::size_t index = 0; index < records_.size(); ++index) {
      const auto& record = records_[index];
      json << "    {\"sequence\": " << record.sequence
           << ", \"layer\": " << quote(record.layer) << ", \"component\": " << quote(record.component)
           << ", \"track\": " << record.track << ", \"frame\": " << record.frame
           << ", \"timestamp\": " << record.timestamp
           << ", \"next_timestamp\": " << record.next_timestamp
           << ", \"dtype\": \"f32le\", \"shape\": [" << record.values << "]"
           << ", \"offset_bytes\": " << record.offset << ", \"byte_length\": " << record.length
           << ", \"sha256\": " << quote(record.hash) << "}"
           << (index + 1 == records_.size() ? "\n" : ",\n");
    }
    json << "  ],\n  \"data_sha256\": "
         << quote(sha256(data_bytes.data(), data_bytes.size())) << "\n}\n";
  }

private:
  fs::path root_;
  std::ofstream data_;
  std::uint64_t offset_{};
  std::vector<Record> records_;
  std::mutex mutex_;
};

struct CallbackData {
  ArtifactWriter* writer{};
  std::vector<std::size_t> frames;
  const char* layer{"postprocess"};
  std::size_t fixed_frame{static_cast<std::size_t>(-1)};
  std::mutex mutex;
};

static bool geometry_callback(void* opaque, const nva2f::IGeometryExecutor::Results& result) {
  auto& data = *static_cast<CallbackData*>(opaque);
  auto frame = data.fixed_frame;
  if (frame == static_cast<std::size_t>(-1)) {
    const std::scoped_lock lock(data.mutex);
    frame = data.frames.at(result.trackIndex)++;
  }
  data.writer->push(data.layer, "skin", result.trackIndex, frame, result.timeStampCurrentFrame,
                    result.timeStampNextFrame, result.skinGeometry);
  data.writer->push(data.layer, "tongue", result.trackIndex, frame, result.timeStampCurrentFrame,
                    result.timeStampNextFrame, result.tongueGeometry);
  data.writer->push(data.layer, "jaw", result.trackIndex, frame, result.timeStampCurrentFrame,
                    result.timeStampNextFrame, result.jawTransform);
  const auto eyes = result.eyesRotation;
  data.writer->push(data.layer, "eyes-right", result.trackIndex, frame, result.timeStampCurrentFrame,
                    result.timeStampNextFrame, eyes.View(0, eyes.Size() / 2));
  data.writer->push(data.layer, "eyes-left", result.trackIndex, frame, result.timeStampCurrentFrame,
                    result.timeStampNextFrame, eyes.View(eyes.Size() / 2, eyes.Size() / 2));
  return true;
}

static bool emotion_callback(void* opaque, const nva2e::IEmotionExecutor::Results& result) {
  auto& data = *static_cast<CallbackData*>(opaque);
  const std::scoped_lock lock(data.mutex);
  const auto frame = data.fixed_frame == static_cast<std::size_t>(-1)
      ? data.frames.at(result.trackIndex)++ : data.fixed_frame;
  data.writer->push(data.layer, "emotion", result.trackIndex, frame, result.timeStampCurrentFrame,
                    result.timeStampNextFrame, result.emotions);
  return true;
}

static void blendshape_host_callback(void* opaque,
                                     const nva2f::IBlendshapeExecutor::HostResults& result,
                                     std::error_code error) {
  if (error) return;
  auto& data = *static_cast<CallbackData*>(opaque);
  auto frame = data.fixed_frame;
  if (frame == static_cast<std::size_t>(-1)) {
    const std::scoped_lock lock(data.mutex);
    frame = data.frames.at(result.trackIndex)++;
  }
  data.writer->push_host(data.layer, "weights", result.trackIndex, frame,
                         result.timeStampCurrentFrame, result.timeStampNextFrame,
                         result.weights.Data(), result.weights.Size());
}

static bool blendshape_device_callback(
    void* opaque, const nva2f::IBlendshapeExecutor::DeviceResults& result) {
  auto& data = *static_cast<CallbackData*>(opaque);
  auto frame = data.fixed_frame;
  if (frame == static_cast<std::size_t>(-1)) {
    const std::scoped_lock lock(data.mutex);
    frame = data.frames.at(result.trackIndex)++;
  }
  data.writer->push(data.layer, "weights", result.trackIndex, frame,
                    result.timeStampCurrentFrame, result.timeStampNextFrame,
                    result.weights);
  return true;
}

static std::vector<float> read_samples(const fs::path& path) {
  const auto bytes = read_bytes(path);
  if (bytes.size() % sizeof(float) != 0) throw std::runtime_error("invalid f32le fixture length");
  std::vector<float> samples(bytes.size() / sizeof(float));
  std::memcpy(samples.data(), bytes.data(), bytes.size());
  return samples;
}

int main(int argc, char** argv) try {
  if (argc != 9) {
    std::cerr << "usage: audio2face3d-cpp-reference <regression|diffusion|emotion> "
                 "<standard|interactive-random|interactive-all|interactive-blendshape-random|interactive-blendshape-all|blendshape-cpu|blendshape-gpu|teeth-standalone> <model.json> "
                 "<samples.f32le> <output> <fp32|fp16> <tracks> <seed>\n";
    return 2;
  }
  const std::string pipeline = argv[1];
  const std::string execution = argv[2];
  const fs::path model = argv[3];
  const fs::path fixture = argv[4];
  const fs::path output = argv[5];
  const std::string precision = argv[6];
  const auto tracks = static_cast<std::size_t>(std::stoull(argv[7]));
  const auto seed = static_cast<std::uint64_t>(std::stoull(argv[8]));
  if (tracks == 0) throw std::runtime_error("tracks must be non-zero");
  check(nva2x::SetCudaDeviceIfNeeded(0), "SetCudaDeviceIfNeeded");
  const auto samples = read_samples(fixture);
  ArtifactWriter writer(output);
  CallbackData callback{&writer, std::vector<std::size_t>(tracks)};

  if (execution == "teeth-standalone") {
    if (pipeline != "regression") {
      throw std::runtime_error("standalone teeth reference currently requires a regression model");
    }
    SdkPtr<nva2f::IRegressionModel::IGeometryModelInfo> info(
        nva2f::ReadRegressionModelInfo(model.string().c_str()));
    if (!info) throw std::runtime_error("regression model info creation failed");
    const auto creation = info->GetExecutorCreationParameters(
        nva2f::IGeometryExecutor::ExecutionOption::All, 30, 1);
    const auto* initialization = creation.initializationTeethParams;
    if (initialization == nullptr) {
      throw std::runtime_error("regression model has no teeth initialization data");
    }
    SdkPtr<nva2x::ICudaStream> stream(nva2x::CreateCudaStream());
    SdkPtr<nva2f::IMultiTrackAnimatorTeeth> animator(nva2f::CreateMultiTrackAnimatorTeeth());
    if (!stream) throw std::runtime_error("SDK CUDA stream creation failed");
    if (!animator) throw std::runtime_error("SDK teeth animator creation failed");
    check(animator->SetCudaStream(stream->Data()), "teeth SetCudaStream");
    check(animator->Init(initialization->params, tracks), "teeth Init");
    check(animator->SetAnimatorData(initialization->data), "teeth SetAnimatorData");
    const auto pose_size = initialization->data.neutralJaw.Size();
    const nva2x::TensorBatchInfo input_info{2, pose_size, pose_size + 3};
    const nva2x::TensorBatchInfo output_info{3, 16, 20};
    std::vector<float> deltas(input_info.stride * tracks);
    for (std::size_t track = 0; track < tracks; ++track) {
      auto parameters = initialization->params;
      if (track % 3 == 1) {
        parameters.lowerTeethStrength = 0.5f;
        parameters.lowerTeethHeightOffset = 0.25f;
        parameters.lowerTeethDepthOffset = -0.5f;
      } else if (track % 3 == 2) {
        parameters.lowerTeethStrength = 2.0f;
        parameters.lowerTeethHeightOffset = -3.0f;
        parameters.lowerTeethDepthOffset = 3.0f;
      }
      check(animator->SetParameters(track, parameters), "teeth SetParameters");
      for (std::size_t index = 0; index < pose_size; ++index) {
        deltas[track * input_info.stride + input_info.offset + index] =
            static_cast<float>((track + 1) * (index % 7 + 1)) * 0.001f;
      }
    }
    SdkPtr<nva2x::IDeviceTensorFloat> input(nva2x::CreateDeviceTensorFloat(
        nva2x::HostTensorFloatConstView(deltas.data(), deltas.size()), stream->Data()));
    SdkPtr<nva2x::IDeviceTensorFloat> device_output(
        nva2x::CreateDeviceTensorFloat(output_info.stride * tracks));
    if (!input || !device_output) throw std::runtime_error("SDK teeth tensor creation failed");
    check(animator->ComputeJawTransform(
              static_cast<nva2x::DeviceTensorFloatConstView>(*input), input_info,
              static_cast<nva2x::DeviceTensorFloatView>(*device_output), output_info),
          "teeth ComputeJawTransform");
    check(stream->Synchronize(), "teeth Synchronize");
    std::vector<float> transforms(output_info.stride * tracks);
    check(nva2x::CopyDeviceToHost(
              nva2x::HostTensorFloatView(transforms.data(), transforms.size()),
              static_cast<nva2x::DeviceTensorFloatConstView>(*device_output)),
          "teeth CopyDeviceToHost");
    for (std::size_t track = 0; track < tracks; ++track) {
      writer.push_host("standalone-teeth", "jaw", track, 0, 0, 0,
                       transforms.data() + track * output_info.stride + output_info.offset,
                       output_info.size);
    }
    writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
    return 0;
  }

  if (pipeline == "emotion") {
    if (execution == "interactive-random" || execution == "interactive-all") {
      if (tracks != 1) throw std::runtime_error("interactive execution requires one track");
      SdkPtr<nva2x::ICudaStream> stream(nva2x::CreateCudaStream());
      SdkPtr<nva2x::IAudioAccumulator> audio(nva2x::CreateAudioAccumulator(60000, 0));
      SdkPtr<nva2e::IClassifierModel::IEmotionModelInfo> info(
          nva2e::ReadClassifierModelInfo(model.string().c_str()));
      if (!stream || !audio || !info) throw std::runtime_error("emotion interactive resources failed");
      const nva2x::IAudioAccumulator* audio_pointer = audio.get();
      nva2e::EmotionExecutorCreationParameters parameters;
      parameters.cudaStream = stream->Data();
      parameters.nbTracks = 1;
      parameters.sharedAudioAccumulators = &audio_pointer;
      auto creation = info->GetExecutorCreationParameters(60000, 30, 1, 0);
      SdkPtr<nva2e::IEmotionInteractiveExecutor> executor(
          nva2e::CreateClassifierEmotionInteractiveExecutor(parameters, creation, 1));
      if (!executor) throw std::runtime_error("emotion interactive executor creation failed");
      check(audio->Accumulate(nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
                             stream->Data()), "audio Accumulate");
      check(audio->Close(), "audio Close");
      check(executor->SetResultsCallback(emotion_callback, &callback), "SetResultsCallback");
      if (execution == "interactive-all") {
        callback.layer = "interactive-all";
        check(executor->ComputeAllFrames(), "ComputeAllFrames");
      } else {
        const auto frame = executor->GetTotalNbFrames() / 2;
        callback.fixed_frame = frame;
        callback.layer = "interactive-random";
        check(executor->ComputeFrame(frame), "ComputeFrame random");
        callback.layer = "interactive-replay";
        check(executor->ComputeFrame(frame), "ComputeFrame replay");
        check(executor->Invalidate(nva2e::IEmotionInteractiveExecutor::kLayerPostProcessing),
              "Invalidate postprocess");
        callback.layer = "interactive-invalidation";
        check(executor->ComputeFrame(frame), "ComputeFrame invalidated");
      }
      check(stream->Synchronize(), "interactive Synchronize");
      writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
      return 0;
    }
    if (execution != "standard") {
      throw std::runtime_error("emotion supports standard execution only");
    }
    SdkPtr<nva2e::IEmotionExecutorBundle> bundle(
        nva2e::ReadClassifierEmotionExecutorBundle(
            tracks, model.string().c_str(), 60000, 30, 1, 0, nullptr));
    if (!bundle) throw std::runtime_error("SDK emotion bundle creation failed");
    check(bundle->GetExecutor().SetResultsCallback(emotion_callback, &callback),
          "SetResultsCallback");
    for (std::size_t track = 0; track < tracks; ++track) {
      auto& preferred = bundle->GetPreferredEmotionAccumulator(track);
      std::vector<float> defaults(preferred.GetEmotionSize(), 0.0f);
      check(preferred.Accumulate(0,
                nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()),
                bundle->GetCudaStream().Data()),
            "preferred emotion Accumulate");
      check(preferred.Close(), "preferred emotion Close");
      check(bundle->GetAudioAccumulator(track).Accumulate(
                nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
                bundle->GetCudaStream().Data()),
            "audio Accumulate");
      check(bundle->GetAudioAccumulator(track).Close(), "audio Close");
    }
    while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
      check(bundle->GetExecutor().Execute(nullptr), "Execute");
    }
    writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
    return 0;
  }

  if (execution == "interactive-blendshape-random" ||
      execution == "interactive-blendshape-all") {
    if (tracks != 1) throw std::runtime_error("interactive execution requires one track");
    SdkPtr<nva2x::ICudaStream> stream(nva2x::CreateCudaStream());
    SdkPtr<nva2x::IAudioAccumulator> audio(nva2x::CreateAudioAccumulator(16000, 0));
    if (!stream || !audio) throw std::runtime_error("interactive accumulator creation failed");

    nva2f::GeometryExecutorCreationParameters parameters;
    parameters.cudaStream = stream->Data();
    parameters.nbTracks = 1;
    const nva2x::IAudioAccumulator* audio_pointer = audio.get();
    parameters.sharedAudioAccumulators = &audio_pointer;
    SdkPtr<nva2x::IEmotionAccumulator> emotion;
    SdkPtr<nva2f::IGeometryInteractiveExecutor> geometry;
    SdkPtr<nva2f::IBlendshapeInteractiveExecutor> executor;

    if (pipeline == "regression") {
      SdkPtr<nva2f::IRegressionModel::IGeometryModelInfo> geometry_info(
          nva2f::ReadRegressionModelInfo(model.string().c_str()));
      SdkPtr<nva2f::IRegressionModel::IBlendshapeSolveModelInfo> blendshape_info(
          nva2f::ReadRegressionBlendshapeSolveModelInfo(model.string().c_str()));
      if (!geometry_info || !blendshape_info) {
        throw std::runtime_error("regression interactive model info creation failed");
      }
      const auto emotion_size = geometry_info->GetNetworkInfo().GetEmotionsCount();
      emotion.reset(nva2x::CreateEmotionAccumulator(emotion_size, 300, 0));
      const nva2x::IEmotionAccumulator* emotion_pointer = emotion.get();
      parameters.sharedEmotionAccumulators = &emotion_pointer;
      const auto geometry_creation = geometry_info->GetExecutorCreationParameters(
          nva2f::IGeometryExecutor::ExecutionOption::All, 30, 1);
      geometry.reset(nva2f::CreateRegressionGeometryInteractiveExecutor(
          parameters, geometry_creation, 0));
      const auto blendshape_creation = blendshape_info->GetExecutorCreationParameters(
          nva2f::IGeometryExecutor::ExecutionOption::All);
      nva2f::DeviceBlendshapeSolveExecutorCreationParameters blendshape_parameters;
      blendshape_parameters.initializationSkinParams =
          blendshape_creation.initializationSkinParams;
      blendshape_parameters.initializationTongueParams =
          blendshape_creation.initializationTongueParams;
      executor.reset(nva2f::CreateDeviceBlendshapeSolveInteractiveExecutor(
          geometry.release(), blendshape_parameters));
      std::vector<float> defaults(emotion_size, 0.0f);
      check(emotion->Accumulate(
                0, nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()),
                stream->Data()),
            "emotion Accumulate");
    } else if (pipeline == "diffusion") {
      SdkPtr<nva2f::IDiffusionModel::IGeometryModelInfo> geometry_info(
          nva2f::ReadDiffusionModelInfo(model.string().c_str()));
      SdkPtr<nva2f::IDiffusionModel::IBlendshapeSolveModelInfo> blendshape_info(
          nva2f::ReadDiffusionBlendshapeSolveModelInfo(model.string().c_str()));
      if (!geometry_info || !blendshape_info) {
        throw std::runtime_error("diffusion interactive model info creation failed");
      }
      const auto emotion_size = geometry_info->GetNetworkInfo().GetEmotionsCount();
      emotion.reset(nva2x::CreateEmotionAccumulator(emotion_size, 300, 0));
      const nva2x::IEmotionAccumulator* emotion_pointer = emotion.get();
      parameters.sharedEmotionAccumulators = &emotion_pointer;
      const auto geometry_creation = geometry_info->GetExecutorCreationParameters(
          nva2f::IGeometryExecutor::ExecutionOption::All, 0, true);
      geometry.reset(nva2f::CreateDiffusionGeometryInteractiveExecutor(
          parameters, geometry_creation, 0));
      const auto blendshape_creation = blendshape_info->GetExecutorCreationParameters(
          nva2f::IGeometryExecutor::ExecutionOption::All, 0);
      nva2f::DeviceBlendshapeSolveExecutorCreationParameters blendshape_parameters;
      blendshape_parameters.initializationSkinParams =
          blendshape_creation.initializationSkinParams;
      blendshape_parameters.initializationTongueParams =
          blendshape_creation.initializationTongueParams;
      executor.reset(nva2f::CreateDeviceBlendshapeSolveInteractiveExecutor(
          geometry.release(), blendshape_parameters));
      std::vector<float> defaults(emotion_size, 0.0f);
      check(emotion->Accumulate(
                0, nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()),
                stream->Data()),
            "emotion Accumulate");
    } else {
      throw std::runtime_error("interactive BlendShape requires a geometry pipeline");
    }
    if (!executor) throw std::runtime_error("interactive BlendShape executor creation failed");
    check(emotion->Close(), "emotion Close");
    check(audio->Accumulate(nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
                            stream->Data()), "audio Accumulate");
    check(audio->Close(), "audio Close");
    check(executor->SetResultsCallback(blendshape_device_callback, &callback),
          "SetResultsCallback");
    const auto target = executor->GetTotalNbFrames() / 2;
    if (execution == "interactive-blendshape-all") {
      callback.layer = "interactive-blendshape-all";
      check(executor->ComputeAllFrames(), "BlendShape ComputeAllFrames");
    } else {
      callback.fixed_frame = target;
      callback.layer = "interactive-blendshape-random";
      check(executor->ComputeFrame(target), "BlendShape ComputeFrame random");
      callback.layer = "interactive-blendshape-replay";
      check(executor->ComputeFrame(target), "BlendShape ComputeFrame replay");
      check(executor->Invalidate(nva2f::IBlendshapeInteractiveExecutor::kLayerBlendshapeWeights),
            "Invalidate BlendShape weights");
      callback.layer = "interactive-blendshape-invalidation";
      check(executor->ComputeFrame(target), "BlendShape ComputeFrame invalidated");
    }
    check(stream->Synchronize(), "interactive BlendShape Synchronize");
    writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
    return 0;
  }

  if (execution == "interactive-random" || execution == "interactive-all") {
    if (tracks != 1) throw std::runtime_error("interactive execution requires one track");
    SdkPtr<nva2x::ICudaStream> stream(nva2x::CreateCudaStream());
    SdkPtr<nva2x::IAudioAccumulator> audio(nva2x::CreateAudioAccumulator(16000, 0));
    if (!stream || !audio) throw std::runtime_error("interactive accumulator creation failed");

    nva2f::GeometryExecutorCreationParameters parameters;
    parameters.cudaStream = stream->Data();
    parameters.nbTracks = 1;
    const nva2x::IAudioAccumulator* audio_pointer = audio.get();
    parameters.sharedAudioAccumulators = &audio_pointer;
    std::size_t emotion_size = 0;
    SdkPtr<nva2x::IEmotionAccumulator> emotion;
    SdkPtr<nva2f::IGeometryInteractiveExecutor> executor;

    if (pipeline == "regression") {
      SdkPtr<nva2f::IRegressionModel::IGeometryModelInfo> info(
          nva2f::ReadRegressionModelInfo(model.string().c_str()));
      if (!info) throw std::runtime_error("regression model info creation failed");
      emotion_size = info->GetNetworkInfo().GetEmotionsCount();
      emotion.reset(nva2x::CreateEmotionAccumulator(emotion_size, 300, 0));
      const nva2x::IEmotionAccumulator* emotion_pointer = emotion.get();
      parameters.sharedEmotionAccumulators = &emotion_pointer;
      const auto creation = info->GetExecutorCreationParameters(
          nva2f::IGeometryExecutor::ExecutionOption::All, 30, 1);
      executor.reset(nva2f::CreateRegressionGeometryInteractiveExecutor(parameters, creation, 0));
      if (!executor) throw std::runtime_error("regression interactive executor creation failed");
      std::vector<float> defaults(emotion_size, 0.0f);
      check(emotion->Accumulate(0,
                nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()), stream->Data()),
            "emotion Accumulate");
      check(emotion->Close(), "emotion Close");
      check(audio->Accumulate(nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
                              stream->Data()), "audio Accumulate");
      check(audio->Close(), "audio Close");
      check(executor->SetResultsCallback(geometry_callback, &callback), "SetResultsCallback");
      const auto target = executor->GetTotalNbFrames() / 2;
      if (execution == "interactive-all") {
        callback.layer = "interactive-all";
        check(executor->ComputeAllFrames(), "ComputeAllFrames");
      } else {
        callback.fixed_frame = target;
        callback.layer = "interactive-random";
        check(executor->ComputeFrame(target), "ComputeFrame random");
        callback.layer = "interactive-replay";
        check(executor->ComputeFrame(target), "ComputeFrame replay");
        check(executor->Invalidate(nva2f::IGeometryInteractiveExecutor::kLayerSkin),
              "Invalidate skin");
        callback.layer = "interactive-invalidation";
        check(executor->ComputeFrame(target), "ComputeFrame invalidated");
      }
    } else if (pipeline == "diffusion") {
      SdkPtr<nva2f::IDiffusionModel::IGeometryModelInfo> info(
          nva2f::ReadDiffusionModelInfo(model.string().c_str()));
      if (!info) throw std::runtime_error("diffusion model info creation failed");
      emotion_size = info->GetNetworkInfo().GetEmotionsCount();
      emotion.reset(nva2x::CreateEmotionAccumulator(emotion_size, 300, 0));
      const nva2x::IEmotionAccumulator* emotion_pointer = emotion.get();
      parameters.sharedEmotionAccumulators = &emotion_pointer;
      const auto creation = info->GetExecutorCreationParameters(
          nva2f::IGeometryExecutor::ExecutionOption::All, 0, true);
      executor.reset(nva2f::CreateDiffusionGeometryInteractiveExecutor(parameters, creation, 0));
      if (!executor) throw std::runtime_error("diffusion interactive executor creation failed");
      std::vector<float> defaults(emotion_size, 0.0f);
      check(emotion->Accumulate(0,
                nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()), stream->Data()),
            "emotion Accumulate");
      check(emotion->Close(), "emotion Close");
      check(audio->Accumulate(nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
                              stream->Data()), "audio Accumulate");
      check(audio->Close(), "audio Close");
      check(executor->SetResultsCallback(geometry_callback, &callback), "SetResultsCallback");
      const auto target = executor->GetTotalNbFrames() / 2;
      if (execution == "interactive-all") {
        callback.layer = "interactive-all";
        check(executor->ComputeAllFrames(), "ComputeAllFrames");
      } else {
        callback.fixed_frame = target;
        callback.layer = "interactive-random";
        check(executor->ComputeFrame(target), "ComputeFrame random");
        callback.layer = "interactive-replay";
        check(executor->ComputeFrame(target), "ComputeFrame replay");
        check(executor->Invalidate(nva2f::IGeometryInteractiveExecutor::kLayerSkin),
              "Invalidate skin");
        callback.layer = "interactive-invalidation";
        check(executor->ComputeFrame(target), "ComputeFrame invalidated");
      }
    } else {
      throw std::runtime_error("interactive execution requires a geometry pipeline");
    }
    check(stream->Synchronize(), "interactive Synchronize");
    writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
    return 0;
  }

  if (execution == "blendshape-cpu" || execution == "blendshape-gpu") {
    const bool gpu = execution == "blendshape-gpu";
    nva2f::IBlendshapeExecutorBundle* raw_blendshape = nullptr;
    if (pipeline == "regression") {
      raw_blendshape = nva2f::ReadRegressionBlendshapeSolveExecutorBundle(
          tracks, model.string().c_str(), nva2f::IGeometryExecutor::ExecutionOption::All,
          gpu, 30, 1, nullptr, nullptr);
    } else if (pipeline == "diffusion") {
      raw_blendshape = nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
          tracks, model.string().c_str(), nva2f::IGeometryExecutor::ExecutionOption::All,
          gpu, 0, true, nullptr, nullptr);
    } else {
      throw std::runtime_error("blendshape requires a geometry pipeline");
    }
    if (raw_blendshape == nullptr) throw std::runtime_error("SDK blendshape bundle creation failed");
    SdkPtr<nva2f::IBlendshapeExecutorBundle> bundle(raw_blendshape);
    callback.layer = "blendshape";
    if (gpu) {
      check(bundle->GetExecutor().SetResultsCallback(blendshape_device_callback, &callback),
            "SetResultsCallback");
    } else {
      check(bundle->GetExecutor().SetResultsCallback(blendshape_host_callback, &callback),
            "SetResultsCallback");
    }
    for (std::size_t track = 0; track < tracks; ++track) {
      auto& emotion = bundle->GetEmotionAccumulator(track);
      std::vector<float> defaults(emotion.GetEmotionSize(), 0.0f);
      check(emotion.Accumulate(0,
                nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()),
                bundle->GetCudaStream().Data()),
            "emotion Accumulate");
      check(emotion.Close(), "emotion Close");
      check(bundle->GetAudioAccumulator(track).Accumulate(
                nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
                bundle->GetCudaStream().Data()),
            "audio Accumulate");
      check(bundle->GetAudioAccumulator(track).Close(), "audio Close");
    }
    while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
      check(bundle->GetExecutor().Execute(nullptr), "Execute");
    }
    for (std::size_t track = 0; track < tracks; ++track) {
      check(bundle->GetExecutor().Wait(track), "Wait");
    }
    writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
    return 0;
  }
  if (execution != "standard") {
    throw std::runtime_error("unsupported execution mode");
  }

  nva2f::IGeometryExecutorBundle* raw = nullptr;
  if (pipeline == "regression") {
    raw = nva2f::ReadRegressionGeometryExecutorBundle(
        tracks, model.string().c_str(), nva2f::IGeometryExecutor::ExecutionOption::All,
        30, 1, nullptr);
  } else if (pipeline == "diffusion") {
    raw = nva2f::ReadDiffusionGeometryExecutorBundle(
        tracks, model.string().c_str(), nva2f::IGeometryExecutor::ExecutionOption::All,
        0, true, nullptr);
  } else {
    throw std::runtime_error("pipeline must be regression, diffusion, or emotion");
  }
  if (raw == nullptr) throw std::runtime_error("SDK bundle creation failed");
  SdkPtr<nva2f::IGeometryExecutorBundle> bundle(raw);
  check(bundle->GetExecutor().SetResultsCallback(geometry_callback, &callback),
        "SetResultsCallback");
  for (std::size_t track = 0; track < tracks; ++track) {
    auto& emotion = bundle->GetEmotionAccumulator(track);
    std::vector<float> defaults(emotion.GetEmotionSize(), 0.0f);
    check(emotion.Accumulate(0,
              nva2x::HostTensorFloatConstView(defaults.data(), defaults.size()),
              bundle->GetCudaStream().Data()),
          "emotion Accumulate");
    check(emotion.Close(), "emotion Close");
    check(bundle->GetAudioAccumulator(track).Accumulate(
              nva2x::HostTensorFloatConstView(samples.data(), samples.size()),
              bundle->GetCudaStream().Data()),
          "audio Accumulate");
    check(bundle->GetAudioAccumulator(track).Close(), "audio Close");
  }
  while (nva2x::GetNbReadyTracks(bundle->GetExecutor()) > 0) {
    check(bundle->GetExecutor().Execute(nullptr), "Execute");
  }
  writer.finish(pipeline, execution, precision, seed, tracks, fixture, model);
  return 0;
} catch (const std::exception& error) {
  std::cerr << "audio2face3d-cpp-reference: " << error.what() << '\n';
  return 1;
}
