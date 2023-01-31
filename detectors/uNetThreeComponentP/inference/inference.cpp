#include <string>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "uussmlmodels/detectors/uNetThreeComponentP/inference.hpp"
#include "utilities.hpp"

#define SAMPLING_RATE 100
#define MINIMUM_SIGNAL_LENGTH 1008
#define EXPECTED_SIGNAL_LENGTH 1008

using namespace UUSSMLModels::Detectors::UNetThreeComponentP;
#include "openvino.hpp"

class Inference::InferenceImpl
{
public:
    OpenVINOImpl mOpenVINO;   
    bool mUseOpenVINO{false};
    bool mInitialized{false};
};

/// Constructor
Inference::Inference() :
    pImpl(std::make_unique<InferenceImpl> ())
{
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
bool Inference::isValidSignalLength(const int nSamples) const noexcept
{
    if (nSamples < getMinimumSignalLength()){return false;}
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
        throw std::runtime_error("Recompiled with OpenVino to read ONNX");
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
    Inference::predictProbability(const std::vector<U> &vertical,
                                  const std::vector<U> &north,
                                  const std::vector<U> &east) const
{   
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (vertical.size() != north.size() || vertical.size() != east.size())
    {
        throw std::invalid_argument("Inconsistent signal sizes");
    }
    if (static_cast<int> (vertical.size()) != getExpectedSignalLength())
    {
        throw std::invalid_argument("Signal length must equal "
                                  + std::to_string(getExpectedSignalLength()));
    }
    std::vector<U> probabilitySignal;
    if (pImpl->mUseOpenVINO)
    {
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.predictProbability(vertical, north, east, &probabilitySignal);
#else
       throw std::runtime_error("Recompile with OpenVino");
#endif
    }
    return probabilitySignal;
}

template<typename U>
std::vector<U>
    Inference::predictProbabilitySlidingWindow(const std::vector<U> &vertical,
                                               const std::vector<U> &north,
                                               const std::vector<U> &east) const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (vertical.size() != north.size() || vertical.size() != east.size())
    {
        throw std::invalid_argument("Inconsistent signal sizes");
    }
    if (static_cast<int> (vertical.size()) < getExpectedSignalLength())
    {
        throw std::invalid_argument("Signal length must be at least "
                                  + std::to_string(getExpectedSignalLength()));
    }
    std::vector<U> probabilitySignal;
    constexpr int windowStart = 254;
    constexpr int windowEnd   = 754;
#ifndef NDEBUG
    assert(windowEnd - windowStart == 500);
#endif
    if (pImpl->mUseOpenVINO)
    {   
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.predictProbabilitySlidingWindow(vertical, north, east,
                                                         &probabilitySignal,
                                                         windowStart,
                                                         windowEnd);
#else
       throw std::runtime_error("Recompile with OpenVino");
#endif
    }

    return probabilitySignal;
}


///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template std::vector<double>
UUSSMLModels::Detectors::UNetThreeComponentP::Inference::predictProbability(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &) const;
template std::vector<float> 
UUSSMLModels::Detectors::UNetThreeComponentP::Inference::predictProbability(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &) const;
template std::vector<double> 
UUSSMLModels::Detectors::UNetThreeComponentP::Inference::predictProbabilitySlidingWindow(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &) const;
template std::vector<float> 
UUSSMLModels::Detectors::UNetThreeComponentP::Inference::predictProbabilitySlidingWindow(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &) const;
