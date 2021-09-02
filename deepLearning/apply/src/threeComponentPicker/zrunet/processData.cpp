#include <iostream>
#include <cmath>
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#include "uuss/threeComponentPicker/zrunet/processData.hpp"

using namespace UUSS::ThreeComponentPicker::ZRUNet;

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
    const bool mZeroPhase = false; // Filter has to be causal for picking
    /// Will be closer to a real-time implementation
    //RTSeis::PostProcessing::SingleChannel::InterpolationMethod  mInterpolationMethod
    //    = RTSeis::PostProcessing::SingleChannel::InterpolationMethod::WEIGHTED_AVERAGE_SLOPES;
    RTSeis::PostProcessing::SingleChannel::InterpolationMethod  mInterpolationMethod
       = RTSeis::PostProcessing::SingleChannel::InterpolationMethod::DFT;//WEIGHTED_AVERAGE_SLOPES;
}; 

/// Constructor
ProcessData::ProcessData() :
    pImpl(std::make_unique<ProcessDataImpl> ())
{
}

/// Move c'tor
ProcessData::ProcessData(ProcessData &&process) noexcept
{
    *this = std::move(process);
}

/// Move assignment
ProcessData& ProcessData::operator=(ProcessData &&process) noexcept
{
    if (&process == this){return *this;}
    pImpl = std::move(process.pImpl);
    return *this;
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

/// Processes waveforms
template<typename U>
std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
ProcessData::processWaveforms(
    const std::tuple<const std::vector<U> &,
                     const std::vector<U> &,
                     const std::vector<U> &> &waveforms, 
    const double samplingPeriod)
{
    if (samplingPeriod <= 0)
    {
        throw std::invalid_argument("Sampling period must be positive");
    }
    auto z = std::get<0> (waveforms); 
    auto n = std::get<1> (waveforms);
    auto e = std::get<2> (waveforms);
    if (z.empty()){throw std::invalid_argument("z is empty");}
    if (n.empty()){throw std::invalid_argument("n is empty");}
    if (e.empty()){throw std::invalid_argument("e is empty");}
    if (z.size() != n.size() || z.size() != e.size())
    {
        std::cerr << "Warning - inconsistent sizes" << std::endl;
    }
    std::vector<U> zp; 
    std::vector<U> np; 
    std::vector<U> ep; 
    processWaveform(z.size(), samplingPeriod, z.data(), &zp);
    processWaveform(n.size(), samplingPeriod, n.data(), &np);
    processWaveform(e.size(), samplingPeriod, e.data(), &ep);
    return std::tuple(zp, np, ep);
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
        throw std::invalid_argument("High corner exceeds nyquist");
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
double ProcessData::getTargetSamplingPeriod() const noexcept
{
    return pImpl->mTargetSamplingPeriod;
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
UUSS::ThreeComponentPicker::ZRUNet::ProcessData::processWaveforms(
    const std::tuple<const std::vector<double> &,
                     const std::vector<double> &,
                     const std::vector<double> &> &waveforms,
    double samplingPeriod);

template
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
UUSS::ThreeComponentPicker::ZRUNet::ProcessData::processWaveforms(
    const std::tuple<const std::vector<float> &,
                     const std::vector<float> &,
                     const std::vector<float> &> &waveforms,
    double samplingPeriod);

