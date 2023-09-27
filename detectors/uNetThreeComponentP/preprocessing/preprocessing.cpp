#include <iostream>
#include <cmath>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#endif
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#include "uussmlmodels/detectors/uNetThreeComponentP/preprocessing.hpp"

#define TARGET_SAMPLING_RATE 100
#define TARGET_SAMPLING_PERIOD 0.01

using namespace UUSSMLModels::Detectors::UNetThreeComponentP;

class Preprocessing::PreprocessingImpl
{
public:
    void processWaveform(const std::vector<float> &data,
                         const double samplingPeriod,
                         std::vector<float> *processedData)
    {
 #ifndef NDEBUG
        assert(!data.empty());
        assert(samplingPeriod > 0);
        assert(processedData != nullptr);
        assert(workSpace != nullptr);
#endif
        auto nSamples = data.size();
        std::vector<double> dData(nSamples);
        std::copy(data.begin(), data.end(), dData.begin());
        std::vector<double> workSpace;
        processWaveform(dData, samplingPeriod, &workSpace);
        int nSamplesNew = static_cast<int> (workSpace.size());
        processedData->resize(nSamplesNew);
        std::copy(workSpace.begin(), workSpace.end(), processedData->begin());
    }
    void processWaveform(const std::vector<double> &data,
                         const double samplingPeriod,
                         std::vector<double> *processedData) 
    {
        auto nyquistFrequency = 1/(2*samplingPeriod);
        if (nyquistFrequency < mCorners.second)
        {
            throw std::invalid_argument("High corner exceeds Nyquist");
        }
#ifndef NDEBUG
        assert(!data.empty());
        assert(samplingPeriod > 0);
        assert(processedData != nullptr);
#endif
        // Set the data
        mWave.setData(data);
        mWave.setSamplingPeriod(samplingPeriod);
        // Process the data
        // (1) Remove trend
        if (mRemoveTrend){mWave.detrend();}
        // (2) Taper to minimize edge effects near filter start
        if (mDoTaper){mWave.taper(mTaperPct);}
        // (3) Filter: Note that the bandpass will prevent aliasing
        //     when downsampling the 200 Hz stations.
        mWave.iirBandpassFilter(mFilterPoles,
                                mCorners,
                                mPrototype,
                                mRipple,
                                mZeroPhase);
        // (4) Potentially resample to target sampling period (e.g., 100 Hz)
        if (std::abs(samplingPeriod - mTargetSamplingPeriod) > 1.e-5)
        {
           mWave.interpolate(mTargetSamplingPeriod, mInterpolationMethod);
        }
        // Get the data
        auto nNewSamples = mWave.getOutputLength();
        processedData->resize(nNewSamples);
        double *dataPtr = processedData->data();
        mWave.getData(nNewSamples, &dataPtr);
    }
///private:
    /// Holds the data processing waveform
    RTSeis::PostProcessing::SingleChannel::Waveform<double> mWave;
    /// Anticipating 100 Hz sampling rate
    const double mTargetSamplingPeriod{TARGET_SAMPLING_PERIOD};
    /// Remove trend
    const bool mRemoveTrend{true};
    /// Taper the data prior to filtering
    const double mTaperPct{1};
    const bool mDoTaper{true};
    /// Bandpass filter data
    const int mFilterPoles{2};
    std::pair<double, double> mCorners = {1, 17}; // 1 Hz to 17 Hz
    const RTSeis::PostProcessing::SingleChannel::IIRPrototype mPrototype{
        RTSeis::PostProcessing::SingleChannel::IIRPrototype::BUTTERWORTH
    };
    const double mRipple{5}; // Ripple for Chebyshev 1 or 2 - not applicable
    const bool mZeroPhase = false; // Filter has to be causal for picking
    /// Will be closer to a real-time implementation
    //RTSeis::PostProcessing::SingleChannel::InterpolationMethod  mInterpolationMethod
    //    = RTSeis::PostProcessing::SingleChannel::InterpolationMethod::WEIGHTED_AVERAGE_SLOPES;
    const RTSeis::PostProcessing::SingleChannel::InterpolationMethod
        mInterpolationMethod
    {
        RTSeis::PostProcessing::SingleChannel::InterpolationMethod::DFT
    };
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

/// Destructor
Preprocessing::~Preprocessing() = default;

/// Reset class
void Preprocessing::clear() noexcept
{
    pImpl = std::make_unique<PreprocessingImpl> ();
}

/// Processes waveforms
template<typename U>
std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
Preprocessing::process(const std::vector<U> &vertical,
                       const std::vector<U> &north,
                       const std::vector<U> &east,
                       const double samplingRate)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling rate must be positive");
    }
    double samplingPeriod = 1./samplingRate;
    if (vertical.empty())
    {
        throw std::invalid_argument("Vertical channel is empty");
    }
    if (north.empty())
    {
        throw std::invalid_argument("North channel is empty");
    }
    if (east.empty())
    {
        throw std::invalid_argument("East channel is empty");
    }
    if (vertical.size() != north.size() ||
        vertical.size() != east.size())
    {
        throw std::invalid_argument("Channel sizes are inconsistent");
    }
    std::vector<U> verticalProcessed;
    std::vector<U> northProcessed;
    std::vector<U> eastProcessed;
    pImpl->processWaveform(vertical, samplingPeriod, &verticalProcessed);
    pImpl->processWaveform(north,    samplingPeriod, &northProcessed);
    pImpl->processWaveform(east,     samplingPeriod, &eastProcessed);
    return std::tuple(verticalProcessed, northProcessed, eastProcessed);
}

/// Target sampling rate/period
double Preprocessing::getTargetSamplingRate() noexcept
{
    return TARGET_SAMPLING_RATE;
}

double Preprocessing::getTargetSamplingPeriod() noexcept
{
    return TARGET_SAMPLING_PERIOD;
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
UUSSMLModels::Detectors::UNetThreeComponentP::Preprocessing::process(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &,
    double );

template
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
UUSSMLModels::Detectors::UNetThreeComponentP::Preprocessing::process(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &,
    double );
