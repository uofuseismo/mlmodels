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
    return MINIMUM_SIGNAL_LENGTH;
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

/// Predict probability
std::vector<float>
    Inference::predictProbability(const std::vector<float> &vertical,
                                  const std::vector<float> &north,
                                  const std::vector<float> &east) const
{   
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    std::vector<float> probabilitySignal;
    if (pImpl->mUseOpenVINO)
    {
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.predictProbability(vertical, north, east, &probabilitySignal);
#else
       throw std::runtime_error("Recompiled with OpenVino to read ONNX");
#endif
    } 
    return probabilitySignal;
}
