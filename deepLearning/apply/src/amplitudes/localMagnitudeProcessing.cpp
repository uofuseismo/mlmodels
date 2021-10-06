#include <string>
#include <vector>
#include <cmath>
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#include <rtseis/amplitude/timeDomainWoodAnderson.hpp>
#include <rtseis/amplitude/timeDomainWoodAndersonParameters.hpp>
//#include <rtseis/filterImplementations/taper.hpp>
//#include <rtseis/filterImplementations/detrend.hpp>
#include <rtseis/utilities/interpolation/interpolate.hpp>
#include "uuss/amplitudes/localMagnitudeProcessing.hpp"

using namespace UUSS::Amplitudes;

namespace
{
bool isVelocityChannel(const std::string &channel)
{
    if (channel.size() < 3)
    {
        throw std::invalid_argument("Channel size must be 3");
    }
    if (channel[1] == 'H' || channel[1] == 'h')
    {
        return true; //isVelocity = true;
    }
    else if (channel[1] == 'N' || channel[1] == 'n')
    {
        return false; //isVelocity = false;
    }
    else
    {
        throw std::invalid_argument("Unhandled channel: " + channel);
    }
}
}

class LocalMagnitudeProcessing::LocalMagnitudeProcessingImpl
{
public:
    /// Holds the data processing waveform
    RTSeis::PostProcessing::SingleChannel::Waveform<double> mWave;
    const double mTaperPercentage = 5;
    const double mTargetSamplingPeriod = 0.01;
    const RTSeis::Amplitude::DetrendType mDemeanType =
        RTSeis::Amplitude::DetrendType::RemoveMean;
    const RTSeis::Amplitude::WindowType mWindowType = 
        RTSeis::Amplitude::WindowType::Sine;
    const RTSeis::Amplitude::HighPassFilter mHighPassFilterVelocity =
        RTSeis::Amplitude::HighPassFilter::No;
    const RTSeis::Amplitude::HighPassFilter mHighPassFilterAcceleration =
        RTSeis::Amplitude::HighPassFilter::Yes;
    const double mQ = 0.998; // RC highpass filter q
    const bool mDetrend = true;
};

/// C'tor
LocalMagnitudeProcessing::LocalMagnitudeProcessing() :
    pImpl(std::make_unique<LocalMagnitudeProcessingImpl> ())
{
}

/// Target sampling period
double LocalMagnitudeProcessing::getTargetSamplingPeriod() const noexcept
{
    return pImpl->mTargetSamplingPeriod;
}

/// Destructor
LocalMagnitudeProcessing::~LocalMagnitudeProcessing() = default;

/// Processes the data
void LocalMagnitudeProcessing::processWaveform(
    const bool isVelocity, const double gain,
    const int npts,
    const double samplingPeriod,
    const float data[],
    std::vector<float> *processedData)
{
    std::vector<double> dataIn(npts);
    std::copy(data, data + npts, dataIn.begin());
    std::vector<double> temp;
    processWaveform(isVelocity, gain, npts, samplingPeriod, dataIn.data(), &temp);
    int nptsNew = static_cast<int> (temp.size());
    processedData->resize(nptsNew, 0);
    const auto tPtr = temp.data();
    auto dPtr = processedData->data();
    std::copy(tPtr, tPtr + nptsNew, dPtr);
}

void LocalMagnitudeProcessing::processWaveform(
    const std::string &channel, const double gain,
    const int npts,
    const double samplingPeriod,
    const float data[],
    std::vector<float> *processedData)
{
    if (channel.size() != 3)
    {
        throw std::invalid_argument("Channel must have length 3");
    }
    bool isVelocity = isVelocityChannel(channel);
    processWaveform(isVelocity, gain, npts, samplingPeriod,
                    data, processedData);
}

void LocalMagnitudeProcessing::processWaveform(
    const std::string &channel, const double gain,
    const int npts,
    const double samplingPeriod,
    const double data[],
    std::vector<double> *processedData)
{
    if (channel.size() != 3)
    {
        throw std::invalid_argument("Channel must have length 3");
    }
    bool isVelocity = isVelocityChannel(channel);
    processWaveform(isVelocity, gain, npts, samplingPeriod,
                    data, processedData);
}

/// Processes the data
void LocalMagnitudeProcessing::processWaveform(
    const bool isVelocity, const double gain,
    const int npts,
    const double samplingPeriod,
    const double data[],
    std::vector<double> *processedData)
{
    if (processedData == nullptr)
    {
        throw std::invalid_argument("processedData is NULL");
    }
    if (npts < 1){throw std::invalid_argument("No samples");}
    if (data == nullptr){throw std::invalid_argument("Data is NULL");}
    if (gain == 0){throw std::invalid_argument("Gain is zero");}
    // Filter only valid for a handful of discrete frequencies
    const double samplingRate = 1/samplingPeriod;
    RTSeis::Amplitude::TimeDomainWoodAndersonParameters parameters;
    if (!parameters.isSamplingRateSupported(samplingRate))
    {
        throw std::invalid_argument("Unsupported sampling rate: "
                                  + std::to_string(samplingRate));
    }
    parameters.setSamplingRate(samplingRate);
    // Follow the workflow for Jiggle
    parameters.setDetrendType(pImpl->mDemeanType);
    parameters.setTaper(pImpl->mTaperPercentage, pImpl->mWindowType);
    if (isVelocity)
    {
        parameters.setHighPassRCFilter(pImpl->mQ,
                                       pImpl->mHighPassFilterVelocity);
    }
    else
    {
        parameters.setHighPassRCFilter(pImpl->mQ,
                                       pImpl->mHighPassFilterAcceleration);
    }
    // Filter it
    processedData->resize(npts);
    RTSeis::Amplitude::TimeDomainWoodAnderson
        <RTSeis::ProcessingMode::POST, double> woodAnderson;
    woodAnderson.initialize(parameters);
    auto yPtr = processedData->data();
    woodAnderson.apply(npts, data, &yPtr);
    // Do we have to resample?  The Wood Anderson filter acts a lot like a
    // low-pass filter so I'll go with a simple downsampler.   Just need a
    // downsampling factor.
    if (std::abs(samplingPeriod - pImpl->mTargetSamplingPeriod) > 1.e-4)
    {
        auto ratio = samplingPeriod/pImpl->mTargetSamplingPeriod;
        auto nPointsNew = static_cast<int> (std::round(npts*ratio));
        std::vector<double> xWork(*processedData);
        processedData->resize(nPointsNew, 0);
        RTSeis::Utilities::Interpolation::interpft(npts, xWork.data(),
                                                   nPointsNew, &yPtr);
    }
}
