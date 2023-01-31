#include "uussmlmodels/detectors/uNetThreeComponentS/inference.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentP/inference.hpp"

using namespace UUSSMLModels::Detectors::UNetThreeComponentS;
namespace PModel = UUSSMLModels::Detectors::UNetThreeComponentP;

class Inference::InferenceImpl
{
public:
    explicit InferenceImpl(const Inference::Device device) :
        mInference(static_cast<PModel::Inference::Device> (device)) 
    {
    }
    PModel::Inference mInference;
};

/// Constructor
Inference::Inference() :
    pImpl(std::make_unique<InferenceImpl> (Inference::Device::CPU))
{
}

/// Destructor
Inference::~Inference() = default;

/// Sampling rate
double Inference::getSamplingRate() noexcept
{ 
    return PModel::Inference::getSamplingRate();
}

/// Minimum signal length
int Inference::getMinimumSignalLength() noexcept
{
    return PModel::Inference::getMinimumSignalLength();
}

/// Expected signal length
int Inference::getExpectedSignalLength() noexcept
{
    return PModel::Inference::getExpectedSignalLength();
}

/// Valid?
bool Inference::isValidSignalLength(const int nSamples) const noexcept
{
    return pImpl->mInference.isValidSignalLength(nSamples);
}

/// Predict probability
template<typename U>
std::vector<U>
    Inference::predictProbability(const std::vector<U> &vertical,
                                  const std::vector<U> &north,
                                  const std::vector<U> &east) const
{
    return pImpl->mInference.predictProbability(vertical, north, east);
}

/// Predict probability
template<typename U>
std::vector<U>
    Inference::predictProbabilitySlidingWindow(const std::vector<U> &vertical,
                                               const std::vector<U> &north,
                                               const std::vector<U> &east) const
{
    return pImpl->mInference.predictProbabilitySlidingWindow(vertical,
                                                             north,
                                                             east);
}

///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template std::vector<double>
UUSSMLModels::Detectors::UNetThreeComponentS::Inference::predictProbability(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &) const;
template std::vector<float>
UUSSMLModels::Detectors::UNetThreeComponentS::Inference::predictProbability(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &) const;
template std::vector<double>
UUSSMLModels::Detectors::UNetThreeComponentS::Inference::predictProbabilitySlidingWindow(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &) const;
template std::vector<float>
UUSSMLModels::Detectors::UNetThreeComponentS::Inference::predictProbabilitySlidingWindow(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &) const;
