#include <cmath>
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/preprocessing.hpp"
 
#define TARGET_SAMPLING_RATE 100
#define TARGET_SAMPLING_PERIOD 0.01

using namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP;

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
        for (int i = 0; i < mCascades; ++i)
        {
            mWave.iirBandpassFilter(mFilterPoles,
                                    mCorners,
                                    mPrototype,
                                    mRipple,
                                    mZeroPhase);
        }
        // (4) Potentially esample to target sampling period (e.g., 100 Hz)
        if (std::abs(samplingPeriod - mTargetSamplingPeriod) > 1.e-5)
        {
            mWave.interpolate(mTargetSamplingPeriod,
                              mInterpolationMethod);
        }
        // Get the data
        auto nNewSamples = mWave.getOutputLength();
        processedData->resize(nNewSamples);
        double *dataPtr = processedData->data();
        mWave.getData(nNewSamples, &dataPtr);
    }
//private:
    /// Holds the data processing waveform
    RTSeis::PostProcessing::SingleChannel::Waveform<double> mWave;
    /// Anticipating 100 Hz sampling rate
    double mTargetSamplingPeriod{TARGET_SAMPLING_PERIOD};
    /// Remove trend
    bool mRemoveTrend{true};
    /// Taper the data prior to filtering
    double mTaperPct{1};
    bool mDoTaper{true};
    /// Bandpass filter data
    int mFilterPoles{2};
    /// Number of cascades
    int mCascades{1};
    std::pair<double, double> mCorners = {1, 17}; // 1 Hz to 17 Hz
    RTSeis::PostProcessing::SingleChannel::IIRPrototype mPrototype
    {
        RTSeis::PostProcessing::SingleChannel::IIRPrototype::BUTTERWORTH
    };
    double mRipple{5}; // Ripple for Chebyshev 1 or 2 - not applicable
    const bool mZeroPhase{false}; // Filter has to be applied causally
    // This is not for real-time
    RTSeis::PostProcessing::SingleChannel::InterpolationMethod
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

/// Move c'tor
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
    pImpl = std::make_unique<PreprocessingImpl> ();
}

/// Destructor
Preprocessing::~Preprocessing() = default;

/// Processes waveforms
template<typename U>
std::vector<U> Preprocessing::process(const std::vector<U> &vertical,
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
    std::vector<U> verticalProcessed;
    pImpl->processWaveform(vertical, samplingPeriod, &verticalProcessed);
    return verticalProcessed;
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
std::vector<double>
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Preprocessing::process(
    const std::vector<double> &,
    double );

template
std::vector<float>
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Preprocessing::process(
    const std::vector<float> &,
    double );
