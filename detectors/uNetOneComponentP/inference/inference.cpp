#include <string>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "uussmlmodels/detectors/uNetOneComponentP/inference.hpp"
#include "utilities.hpp"

#define SAMPLING_RATE 100
#define MINIMUM_SIGNAL_LENGTH 1008
#define EXPECTED_SIGNAL_LENGTH 1008
#define WINDOW_START 254
#define WINDOW_END   754

using namespace UUSSMLModels::Detectors::UNetOneComponentP;
#ifdef WITH_OPENVINO
#include "openvino.hpp"
#endif

class Inference::InferenceImpl
{
public:
    /// Constructor
    explicit InferenceImpl(const Inference::Device device) :
#ifdef WITH_OPENVINO
        mOpenVINO(device),
#endif
        mDevice(device)
    {
    }
#ifdef WITH_OPENVINO
    OpenVINOImpl mOpenVINO;   
#endif
    Inference::Device mDevice{Inference::Device::CPU};
    bool mUseOpenVINO{false};
    bool mInitialized{false};
};

/// Constructor
Inference::Inference() :
    Inference(Inference::Device::CPU)
{
}

/// Constructor with given device
Inference::Inference(const Inference::Device device) :
    pImpl(std::make_unique<InferenceImpl> (device))
{
}

/// Reset class and release memory
void Inference::clear() noexcept
{
    auto device = pImpl->mDevice;
    pImpl = std::make_unique<InferenceImpl> (device);
}

/// Destructor
Inference::~Inference() = default;

/// Sampling rate
double Inference::getSamplingRate() noexcept
{ 
    return SAMPLING_RATE;
}

/// Minimum signal length
int Inference::getMinimumSignalLength() noexcept
{
    return MINIMUM_SIGNAL_LENGTH;
}

/// Expected signal length
int Inference::getExpectedSignalLength() noexcept
{
    return EXPECTED_SIGNAL_LENGTH;
}

/// Valid?
bool Inference::isValidSignalLength(const int nSamples) noexcept
{
    if (nSamples < Inference::getMinimumSignalLength()){return false;}
    if (nSamples%16 != 0){return false;}
    return true;
}

/// Load the model
void Inference::load(const std::string &fileName,
                     const Inference::ModelFormat format)
{
    if (format == Inference::ModelFormat::ONNX)
    {
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.load(fileName);
        pImpl->mUseOpenVINO = true;
        pImpl->mInitialized = true;
#else
        throw std::runtime_error("Recompile with OpenVino to read ONNX");
#endif
    }
    else
    {
        throw std::runtime_error("Unhandled model format");
    }
}

/// Initialized?
bool Inference::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

template<typename U>
std::vector<U>
Inference::predictProbability(const std::vector<U> &vertical) const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (static_cast<int> (vertical.size()) != getExpectedSignalLength())
    {
        throw std::invalid_argument("Signal length must equal "
                                  + std::to_string(getExpectedSignalLength()));
    }
    std::vector<U> probabilitySignal;
    if (pImpl->mUseOpenVINO)
    {
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.predictProbability(vertical, &probabilitySignal);
#else
       throw std::runtime_error("Recompile with OpenVino");
#endif
    }
    return probabilitySignal;
}

template<typename U>
std::vector<U>
Inference::predictProbabilitySlidingWindow(
    const std::vector<U> &vertical) const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (static_cast<int> (vertical.size()) < getExpectedSignalLength())
    {
        throw std::invalid_argument("Signal length must be at least "
                                  + std::to_string(getExpectedSignalLength()));
    }
    std::vector<U> probabilitySignal;
    constexpr int windowStart = WINDOW_START;
    constexpr int windowEnd   = WINDOW_END;
#ifndef NDEBUG
    assert(windowEnd - windowStart == 500);
#endif
    if (pImpl->mUseOpenVINO)
    {   
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.predictProbabilitySlidingWindow(vertical,
                                                         &probabilitySignal,
                                                         windowStart,
                                                         windowEnd);
#else
       throw std::runtime_error("Recompile with OpenVino");
#endif
    }

    return probabilitySignal;
}

std::pair<int, int> Inference::getCentralWindowStartEndIndex() noexcept
{
    return std::pair {WINDOW_START, WINDOW_END};
}

///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template std::vector<double>
UUSSMLModels::Detectors::UNetOneComponentP::Inference::predictProbability(
    const std::vector<double> &) const;
template std::vector<float> 
UUSSMLModels::Detectors::UNetOneComponentP::Inference::predictProbability(
    const std::vector<float> &) const;
template std::vector<double> 
UUSSMLModels::Detectors::UNetOneComponentP::Inference::predictProbabilitySlidingWindow(
    const std::vector<double> &) const;
template std::vector<float> 
UUSSMLModels::Detectors::UNetOneComponentP::Inference::predictProbabilitySlidingWindow(
    const std::vector<float> &) const;
