#include "uussmlmodels/detectors/uNetThreeComponentS/preprocessing.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentP/preprocessing.hpp"

using namespace UUSSMLModels::Detectors::UNetThreeComponentS;
namespace PModel = UUSSMLModels::Detectors::UNetThreeComponentP;

class Preprocessing::PreprocessingImpl
{
public:
    PModel::Preprocessing mPreprocessor;
};

/// Constructor
Preprocessing::Preprocessing() :
    pImpl(std::make_unique<PreprocessingImpl> ())
{
}

/// Move constructor
Preprocessing::Preprocessing(Preprocessing &&process) noexcept
{
    *this = std::move(process);
}

/// Move assignment
Preprocessing& Preprocessing::operator=(Preprocessing &&process) noexcept
{
    if (&process == this){return *this;}
    pImpl = std::move(process.pImpl);
    return *this;
}

/// Reset class
void Preprocessing::clear() noexcept
{
    pImpl->mPreprocessor.clear();
}

/// Destructor
Preprocessing::~Preprocessing() = default;

/// Sampling rate
double Preprocessing::getTargetSamplingRate() noexcept
{
    return PModel::Preprocessing::getTargetSamplingRate();
}

/// The sampling period of the processed signals in seconds
double Preprocessing::getTargetSamplingPeriod() noexcept
{
    return PModel::Preprocessing::getTargetSamplingPeriod();
}

template<typename U>
std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
Preprocessing::process(const std::vector<U> &vertical,
                       const std::vector<U> &north,
                       const std::vector<U> &east,
                       const double samplingRate)
{
    return pImpl->mPreprocessor.process(vertical, north, east, samplingRate);
}

///--------------------------------------------------------------------------///
///                            Template Instantiation                        ///
///--------------------------------------------------------------------------///
template
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
UUSSMLModels::Detectors::UNetThreeComponentS::Preprocessing::process(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &,
    double );

template
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
UUSSMLModels::Detectors::UNetThreeComponentS::Preprocessing::process(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &,
    double );
