#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#include "uuss/firstMotion/fmnet/processData.hpp"

using namespace UUSS::FirstMotion::FMNet;

class ProcessData::ProcessDataImpl
{
public:
    /// Holds the data processing waveform
    RTSeis::PostProcessing::SingleChannel::Waveform<double> mWave;
    /// Anticipating 100 Hz sampling rate
    double mTargetSamplingPeriod = 0.01;
    /// Remove trend
    bool mRemoveTrend = true;
    /// Taper the data prior to filtering
    double mTaperPct = 1;
    bool mDoTaper = true;
    /// Bandpass filter data
    int mFilterPoles = 2;
    std::pair<double, double> mCorners = {1, 17}; // 1 Hz to 17 Hz
    RTSeis::PostProcessing::SingleChannel::IIRPrototype mPrototype
        = RTSeis::PostProcessing::SingleChannel::IIRPrototype::BUTTERWORTH;
    double mRipple = 5; // Ripple for Chebyshev 1 or 2 - not applicable
    const bool mZeroPhase = false; // Filter has to be applied causally
    // This is not for real-time
    RTSeis::PostProcessing::SingleChannel::InterpolationMethod  mInterpolationMethod
        = RTSeis::PostProcessing::SingleChannel::InterpolationMethod::DFT;
};

/// Constructor
ProcessData::ProcessData() :
    pImpl(std::make_unique<ProcessDataImpl> ())
{
}

/// Destructor
ProcessData::~ProcessData() = default;

/// Processes the data
void ProcessData::processWaveform(
    const int npts,
    const double samplingPeriod,
    const float data[],
    std::vector<float> *processedData)
{
    std::vector<double> dataIn(npts);
    #pragma omp simd
    for (int i=0; i<npts; ++i){dataIn[i] = static_cast<double> (data[i]);}
    std::vector<double> temp;
    processWaveform(npts, samplingPeriod, dataIn.data(), &temp);
    int nptsNew = static_cast<int> (temp.size());
    processedData->resize(nptsNew, 0);
    auto dPtr = processedData->data();
    #pragma omp simd
    for (int i=0; i<nptsNew; ++i){dPtr[i] = static_cast<float> (temp[i]);}
}

/// Processes the data
void ProcessData::processWaveform(
    const int npts,
    const double samplingPeriod,
    const double data[],
    std::vector<double> *processedData)
{
    if (samplingPeriod <= 0)
    {
        throw std::invalid_argument("Sampling period = "
                                  + std::to_string(samplingPeriod)
                                  + "must be positive");
    }
    auto fnyq = 1/(2*samplingPeriod);
    if (fnyq < pImpl->mCorners.second)
    {
        throw std::invalid_argument("High corner exceeds Nyquist");
    }
    if (npts < 1 || data == nullptr)
    {
        if (npts < 1){throw std::invalid_argument("No data points");}
        throw std::invalid_argument("Data is NULL");
    }
    // Set the data
    pImpl->mWave.setData(npts, data);
    pImpl->mWave.setSamplingPeriod(samplingPeriod);
    // Process the data
    // (1) Remove trend
    if (pImpl->mRemoveTrend){pImpl->mWave.detrend();}
    // (2) Taper to minimize edge effects near filter start
    if (pImpl->mDoTaper){pImpl->mWave.taper(pImpl->mTaperPct);}
    // (3) Filter: Note that the bandpass will prevent aliasing
    //     when downsampling the 200 Hz stations.
    pImpl->mWave.iirBandpassFilter(pImpl->mFilterPoles,
                                   pImpl->mCorners,
                                   pImpl->mPrototype,
                                   pImpl->mRipple,
                                   pImpl->mZeroPhase);
    // (4) Potentially esample to target sampling period (e.g., 100 Hz)
    if (std::abs(samplingPeriod - pImpl->mTargetSamplingPeriod) > 1.e-5)
    {
        pImpl->mWave.interpolate(pImpl->mTargetSamplingPeriod,
                                 pImpl->mInterpolationMethod);
    }
    // Get the data
    auto nNewSamples = pImpl->mWave.getOutputLength();
    processedData->resize(nNewSamples);
    double *dataPtr = processedData->data();
    pImpl->mWave.getData(nNewSamples, &dataPtr);
}

/// Get sampling period
double ProcessData::getTargetSamplingPeriod() const
{
    return pImpl->mTargetSamplingPeriod;
}

