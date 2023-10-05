#define USE_SPECTROGRAM 1
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#endif
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#ifdef USE_SPECTROGRAM
#include <rtseis/transforms/spectrogram.hpp>
#include <rtseis/transforms/slidingWindowRealDFTParameters.hpp>
#else
#include <rtseis/transforms/wavelets/morlet.hpp>
#include <rtseis/transforms/continuousWavelet.hpp>
#endif
#include "uussmlmodels/eventClassifiers/cnnThreeComponent/preprocessing.hpp"

#define TARGET_SIGNAL_DURATION 45
#define TARGET_SAMPLING_RATE 50
#define SCALOGRAM_SAMPLING_RATE 25
#define TARGET_SAMPLING_PERIOD 0.02
#define SCALOGRAM_SAMPLING_PERIOD 0.04
#define SCALOGRAM_LENGTH 1125
#define N_SCALES 17
#define N_FREQUENCIES 51
#define N_TIME_WINDOWS 72

using namespace UUSSMLModels::EventClassifiers::CNNThreeComponent;

class Preprocessing::PreprocessingImpl
{
public:
    PreprocessingImpl()
    {
#ifdef USE_SPECTROGRAM
        RTSeis::Transforms::SlidingWindowRealDFTParameters parameters;
        parameters.setNumberOfSamples(mTransformSignalLength);
        parameters.setWindow(100,
                             RTSeis::Transforms::SlidingWindowType::HAMMING);
        parameters.setNumberOfSamplesInOverlap(70);
        parameters.setDetrendType(
            RTSeis::Transforms::SlidingWindowDetrendType::REMOVE_MEAN);
        mSpectrogram.initialize(parameters, TARGET_SAMPLING_RATE); 
        mFrequencies = mSpectrogram.getFrequencies();
        mTimeWindows = mSpectrogram.getTimeWindows();
        mNumberOfFrequencies = mSpectrogram.getNumberOfFrequencies();
        mNumberOfTimeWindows = mSpectrogram.getNumberOfTransformWindows();
        mSpectrogramSize = mNumberOfTimeWindows*mNumberOfFrequencies;
 #ifndef NDEBUG
        assert(mNumberOfFrequencies == N_FREQUENCIES);
        assert(mNumberOfTimeWindows == N_TIME_WINDOWS);
 #endif
#else
        // Initialize scalogram
        constexpr double omega0{6}; // tradeoff between time and frequency
        mMorlet.setParameter(omega0);
        mMorlet.enableNormalization();
        mMorlet.disableNormalization();
        // Map the given frequencies to scales for a Morlet wavelet
 #ifndef NDEBUG
        assert(mCenterFrequencies.size() == N_SCALES);
        assert(mScalogramLength == SCALOGRAM_LENGTH);
 #endif
        std::vector<double> scales(mCenterFrequencies.size());
        for (int i = 0; i < static_cast<int> (scales.size()); ++i)
        {
            scales[i] = (omega0*TARGET_SAMPLING_RATE)
                       /(2*M_PI*mCenterFrequencies[i]);
        }
        mCWT.initialize(mTransformSignalLength,
                        scales.size(), scales.data(),
                        mMorlet,
                        TARGET_SAMPLING_RATE); 
        mScalogramSize = mCWT.getNumberOfScales()*mCWT.getNumberOfSamples();
        //std::cout << mTransformSignalLength << " " << mScalogramLength  << " " << mCenterFrequencies.size() << std::endl;
#endif
    }
    void processWaveform(const std::vector<float> &data,
                         const double samplingPeriod,
                         std::vector<float> *processedData)
    {
#ifndef NDEBUG
        assert(!data.empty());
        assert(samplingPeriod > 0); 
        assert(processedData != nullptr);
        assert(workSpace != nullptr);
        assert(mCWT.isInitialized());
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
 #ifdef USE_SPECTROGRAM
        assert(mSpectogram.isInitialized());
 #else
        assert(mCWT.isInitialized());
 #endif
#endif
        // Set the data
        mWave.setData(data);
        mWave.setSamplingPeriod(samplingPeriod);
        // Process the data
        // (1) Remove trend
        if (mRemoveTrend){mWave.detrend();}
        // (2) Taper to minimize edge effects near filter start
        if (mDoTaper)
        {
            mWave.taper(mTaperPct,
                        RTSeis::PostProcessing::SingleChannel::
                                TaperParameters::Type::HANN);
        }
        // (3) Filter: Note that the bandpass will prevent aliasing
        //     when downsampling the 100 Hz stations.
        mWave.iirBandpassFilter(mFilterPoles,
                                mCorners,
                                mPrototype,
                                mRipple,
                                mZeroPhase);
        // (4) Potentially resample to target sampling period (e.g., 50 Hz)
        if (std::abs(samplingPeriod - mTargetSamplingPeriod) > 1.e-5)
        {
            auto downSamplingFactor
                = static_cast<int>
                  (std::round(mTargetSamplingPeriod/samplingPeriod));
            if (std::abs(samplingPeriod*downSamplingFactor
                       - mTargetSamplingPeriod) < 1.e-7 &&
                mTargetSamplingPeriod > samplingPeriod)
            {
                mWave.downsample(downSamplingFactor);
            }
            else
            {
                mWave.interpolate(mTargetSamplingPeriod, mInterpolationMethod);
            }
        }
        // Get the data
        auto workSpace = mWave.getData();
        auto workSpaceLength = static_cast<int> (workSpace.size());
        if (workSpace.empty())
        {
            workSpace.resize(mTransformSignalLength, 0);
        }
        if (workSpaceLength > mTransformSignalLength)
        {
            workSpace.resize(mTransformSignalLength);
        }
        else if (workSpaceLength < mTransformSignalLength)
        {
            workSpace.resize(mTransformSignalLength, 0);
        }
/*
int component = 0;
if (component >= 0)
{
    std::ofstream ofl;
    if (component == 0)
    {
        ofl.open("verticalProc.txt");
    }
    else if (component == 1)
    {
        ofl.open("northProc.txt");
    }
    else if (component == 2)
    {
        ofl.open("eastProc.txt");
    }
    for (int i = 0; i < workSpace.size(); ++i)
    {
        ofl << std::setprecision(12) << i*TARGET_SAMPLING_PERIOD << " " << workSpace[i] << std::endl;
    }
    ofl.close(); 
}
*/
#ifdef USE_SPECTROGRAM
        mSpectrogram.transform(workSpace.size(), workSpace.data());
        // Get the result
        if (static_cast<int> (processedData->size()) != mSpectrogramSize)
        {
            processedData->resize(mSpectrogramSize, 0);
        }
        const double *amplitudePointer = mSpectrogram.getAmplitudePointer();
        std::copy(amplitudePointer, amplitudePointer + mSpectrogramSize,
                  processedData->data());
        double maxAmplitude
            = *std::max_element(processedData->begin(), processedData->end());
        double maxAmplitudeInverse{0};
        if (maxAmplitude > 1.e-8)
        {
            maxAmplitudeInverse = 1./maxAmplitude;
        }
        std::for_each(processedData->begin(), processedData->end(),
                      [&](const double v)
                      {
                          return v*maxAmplitudeInverse;
                      });
#else
        mCWT.transform(workSpace.size(), workSpace.data());
        if (workSpace.size() < mScalogramSize)
        {
            workSpace.resize(mScalogramSize);
        } 
        std::fill(workSpace.begin(), workSpace.end(), 0);
        auto amplitudePointer = workSpace.data();
 #ifndef NDEBUG
        assert(mTransformSignalLength*mCenterFrequencies.size()
               == mScalogramSize);
 #endif
        mCWT.getAmplitudeTransform(mTransformSignalLength,
                                   mCenterFrequencies.size(),
                                   &amplitudePointer);
        // Downsample in time
        if (processedData->size() != mScalogramLength*N_SCALES)
        {
            processedData->resize(mScalogramLength*N_SCALES, 0);
        }
        auto processedDataPointer = processedData->data();
        for (int j = 0; j < N_SCALES; ++j)
        {
            for (int i = 0; i < mScalogramLength; ++i)
            {
                processedDataPointer[j*mScalogramLength + i]
                    = amplitudePointer[j*mTransformSignalLength + 2*i];
            }
        }
        // Normalize
        double maxAmplitude
            = *std::max_element(
                  processedData->begin(),
                  processedData->begin() + mScalogramLength*N_SCALES);
        double maxAmplitudeInverse{0};
        if (maxAmplitude > 1.e-8)
        {
            maxAmplitudeInverse = 1./maxAmplitude;
        }
        std::for_each(processedData->begin(),
                      processedData->begin() + mScalogramLength*N_SCALES,
                      [&](const double v)
                      {
                          return v*maxAmplitudeInverse;
                      });
#endif
/*
 if (component >= 0)
{
    std::ofstream ofl;
    if (component == 0)
    {
        ofl.open("verticalProcAmp.txt");
    }
    else if (component == 1)
    {
        ofl.open("northProcAmp.txt");
    }
    else if (component == 2)
    {
        ofl.open("eastProcAmp.txt");
    }
    for (int j = 0; j < mCenterFrequencies.size(); ++j)
    {
        for (int i = 0; i < mScalogramLength; ++i)
        {
            ofl << std::setprecision(12) << i*SCALOGRAM_SAMPLING_PERIOD << " " 
                << mCenterFrequencies[j] << " " 
                << processedData->at(j*mScalogramLength + i) << std::endl; 
        }
        ofl << std::endl;
    }
    //for (int i = 0; i < mNumberOfTimeWindows; ++i)
    //{
    //    for (int j = 0; j < mNumberOfFrequencies; ++j)
    //    {
    //        int ij = i*mNumberOfFrequencies + j;
    //        ofl << std::setprecision(12) << mTimeWindows[i] << " " << mFrequencies[j] << " " << processedData->at(ij) << std::endl;
    //    }
    //    ofl << std::endl;
    //}
    ofl.close(); 
}
*/
 
    }
///private:
    /// Holds the data processing waveform
    RTSeis::PostProcessing::SingleChannel::Waveform<double> mWave;
#ifdef USE_SPECTROGRAM
    /// Spectrogram calculator
    RTSeis::Transforms::Spectrogram<double> mSpectrogram;
    /// Holds the spectrogram frequencies
    std::vector<double> mFrequencies;
    /// The times of the windows
    std::vector<double> mTimeWindows;
#else
    /// The Morlet wavelet
    RTSeis::Transforms::Wavelets::Morlet mMorlet;
    /// Continuous Wavelet Transform calculator
    RTSeis::Transforms::ContinuousWavelet<double> mCWT;
    /// Holds the center frequencies of the scalogram
    std::vector<double> mCenterFrequencies{ 2,  3,  4,  5,  6,  7,  8,  9, 10,
                                           11, 12, 13, 14, 16, 17, 18, 19};
#endif
    /// Anticipating 50 Hz sampling rate
    double mTargetSamplingPeriod{TARGET_SAMPLING_PERIOD};
    /// Remove trend
    bool mRemoveTrend{true};
    /// Taper the data prior to filtering
    double mTaperPct{1};
    bool mDoTaper{true};
    /// Bandpass filter data
    int mFilterPoles{2};
    /// Number of cascades
    int mCascades{2};
    /// Number of samples in signal to transform
    int mTransformSignalLength
    {
        static_cast<int> (TARGET_SIGNAL_DURATION*TARGET_SAMPLING_RATE) + 1
    };
#ifdef USE_SPECTROGRAM
    /// Number of frequencies in the spectorgram 
    int mNumberOfFrequencies;
    /// Number of time windows
    int mNumberOfTimeWindows;
    /// Number of windows in the spectogram
    int mSpectrogramSize;
#else
    // The number of signals in the output scalogram
    int mScalogramLength{mTransformSignalLength/2};
    /// The scaloegram size
    int mScalogramSize;
#endif
    /// Filter passband
    std::pair<double, double> mCorners = {1, 20}; // 1 Hz to 20 Hz
    RTSeis::PostProcessing::SingleChannel::IIRPrototype mPrototype
    {   
        RTSeis::PostProcessing::SingleChannel::IIRPrototype::BUTTERWORTH
    };
    double mRipple{5}; // Ripple for Chebyshev 1 or 2 - not applicable
    const bool mZeroPhase{true}; // Filter does not need to be applied causally
    // This is not for real-time
    RTSeis::PostProcessing::SingleChannel::InterpolationMethod
        mInterpolationMethod
    {   
        RTSeis::PostProcessing::SingleChannel::InterpolationMethod::WEIGHTED_AVERAGE_SLOPES
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

/// Reset class
void Preprocessing::clear() noexcept
{
    pImpl = std::make_unique<PreprocessingImpl> ();
}

/// Destructor
Preprocessing::~Preprocessing() = default;

/// Process
template<typename U>
std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
Preprocessing::process(
    const std::vector<U> &vertical,
    const std::vector<U> &north,
    const std::vector<U> &east,
    const double samplingRate)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling rate must be positive");
    }
    if (vertical.empty())
    {
        throw std::invalid_argument("Vertical signal is empty");
    }
    if (north.empty())
    {
        throw std::invalid_argument("North signal is empty");
    }
    if (east.empty())
    {
        throw std::invalid_argument("East signal is empty");
    }
#ifdef USE_SPECTOGRAM
    auto verticalSpectrogram = process(vertical, samplingRate);
    auto northSpectrogram    = process(north,    samplingRate);
    auto eastSpectrogram     = process(east,     samplingRate);
    return std::tuple {verticalSpectrogram, northSpectrogram, eastSpectrogram};
#else
    auto verticalScalogram = process(vertical, samplingRate);
    auto northScalogram    = process(north,    samplingRate);
    auto eastScalogram     = process(east,     samplingRate);
    return std::tuple {verticalScalogram, northScalogram, eastScalogram};
#endif
}

/// Process
template<typename U>
std::vector<U> Preprocessing::process(const std::vector<U> &vertical,
                                      const double samplingRate)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling rate must be positive");
    }
    if (vertical.empty())
    {
        throw std::invalid_argument("Vertical signal is empty");
    }
#ifdef USE_SPECTROGRAM
    auto nWork = getSpectrogramSize();
    std::vector<U> verticalSpectrogram(nWork, 0);
    pImpl->processWaveform(vertical, 1./samplingRate, &verticalSpectrogram);
    return verticalSpectrogram;
#else
    auto nWork = getNumberOfScales()*getScalogramLength();
    std::vector<U> verticalScalogram(nWork, 0);
    pImpl->processWaveform(vertical, 1./samplingRate, &verticalScalogram);
    return verticalScalogram;
#endif
}

/// The number of frequencies
int Preprocessing::getNumberOfFrequencies() noexcept
{
    return N_FREQUENCIES;
}

std::vector<double> Preprocessing::getTimeWindows() const
{
    return pImpl->mTimeWindows;
}

std::vector<double> Preprocessing::getFrequencies() const
{
    return pImpl->mFrequencies;
}

int Preprocessing::getSpectrogramSize() noexcept
{
    return N_FREQUENCIES*N_TIME_WINDOWS;
}

int Preprocessing::getNumberOfTimeWindows() noexcept
{
    return N_TIME_WINDOWS;
}

/*
/// Scalogram sampling rate/period
double Preprocessing::getScalogramSamplingRate() noexcept
{
    return SCALOGRAM_SAMPLING_RATE;
}

double Preprocessing::getScalogramSamplingPeriod() noexcept
{
    return SCALOGRAM_SAMPLING_PERIOD;
}

/// The number of scales 
int Preprocessing::getNumberOfScales() noexcept 
{
    return N_SCALES;
}

std::vector<double> Preprocessing::getCenterFrequencies() const
{
    return pImpl->mCenterFrequencies;
}

/// The number of samples in the scalogram
int Preprocessing::getScalogramLength() noexcept
{
    return SCALOGRAM_LENGTH;
}
*/

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template std::vector<double>
UUSSMLModels::EventClassifiers::CNNThreeComponent::Preprocessing::process(
    const std::vector<double> &,
    double );

template std::vector<float>
UUSSMLModels::EventClassifiers::CNNThreeComponent::Preprocessing::process(
    const std::vector<float> &,
    double );

template
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
UUSSMLModels::EventClassifiers::CNNThreeComponent::Preprocessing::process(
    const std::vector<double> &,
    const std::vector<double> &,
    const std::vector<double> &,
    double );

template
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
UUSSMLModels::EventClassifiers::CNNThreeComponent::Preprocessing::process(
    const std::vector<float> &,
    const std::vector<float> &,
    const std::vector<float> &,
    double );

