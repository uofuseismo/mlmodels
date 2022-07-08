#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#endif
#include <rtseis/transforms/continuousWavelet.hpp>
#include <rtseis/transforms/wavelets/morlet.hpp>
#include <rtseis/filterImplementations/detrend.hpp>
#include <rtseis/filterImplementations/taper.hpp>
#include <rtseis/filterImplementations/iirFilter.hpp>
#include <rtseis/filterImplementations/sosFilter.hpp>
#include <rtseis/filterRepresentations/ba.hpp>
#include <rtseis/filterRepresentations/sos.hpp>
#include <rtseis/filterDesign/iir.hpp>
#include <rtseis/utilities/interpolation/weightedAverageSlopes.hpp>
#include "uuss/features/magnitude/channelFeatures.hpp"
#include "uuss/features/magnitude/channel.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"
#include "uuss/features/magnitude/preprocess.hpp"
#include "uuss/features/magnitude/spectralFeatures.hpp"
#include "uuss/features/magnitude/temporalFeatures.hpp"
#include "private/magnitudeUtilities.hpp"

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100
#define PRE_ARRIVAL_TIME -1
#define POST_ARRIVAL_TIME 4
/*
#define TARGET_SIGNAL_LENGTH 500    // 1s before to 4s after
#define PRE_ARRIVAL_TIME 1          // 1s before P arrival
#define POST_ARRIVAL_TIME 4         // 4s after P arrival
#define P_PICK_ERROR 0.05           // Alysha's P pickers are usually within 5 samples
#define TARGET_SIGNAL_DURATION (TARGET_SIGNAL_LENGTH - 1)*TARGET_SAMPLING_PERIOD
*/

using namespace UUSS::Features::Magnitude;

class ChannelFeatures::FeaturesImpl
{
public:
    FeaturesImpl(const std::vector<double> &frequencies,
                 const std::vector<double> &durations,
                 const double preArrivalTime =-1,
                 const double postArrivalTime = 4,
                 const double pickError = 0.05) :
        mCenterFrequencies(frequencies),
        mDurations(durations),
        mPreArrivalTime(preArrivalTime),
        mPostArrivalTime(postArrivalTime),
        mPickError(pickError)
    {
        if (preArrivalTime > 0)
        {
            throw std::invalid_argument("Pre arrival time must negative");
        }
        if (postArrivalTime < 0)
        {
            throw std::invalid_argument("Post arrival time must be positive");
        }
        if (pickError < 0)
        {
            throw std::invalid_argument("Pick error must be non-negative");
        }

        if (mCenterFrequencies.empty())
        {
            throw std::invalid_argument("Frequencies is empty");
        }
        if (mDurations.empty())
        {
            throw std::invalid_argument("Durations is empty");
        }
        for (const auto &c : mCenterFrequencies)
        {
            if (c <= 0)
            {
                throw std::invalid_argument("Invalid center frequency");
            }
        }
        for (const auto &d : mDurations)
        {
            if (d <= 0){throw std::invalid_argument("Invalid duration");}
        }
    }
    // Initialize the CWT
    void initializeCWT()
    {
        auto nSamples = mPreprocess.getTargetSignalLength();
        const double omega0 = 4; // tradeoff between time and frequency
        mMorlet.setParameter(omega0);
        mMorlet.enableNormalization();
        mMorlet.disableNormalization();
        // Map the given frequencies to scales for a Morlet wavelet
        std::vector<double> scales(mCenterFrequencies.size());
        for (int i = 0; i < static_cast<int> (scales.size()); ++i)
        {
            scales[i] = (omega0*mSamplingRate)/(2*M_PI*mCenterFrequencies[i]);
        }
        mCWT.initialize(nSamples,
                        scales.size(), scales.data(),
                        mMorlet,
                        mTargetSamplingRate);
        mAmplitudeCWT.resize(scales.size()*nSamples, 0);
        //mCumulativeAmplitudeCWT.resize(mAmplitudeCWT.size(), 0);
    }
    // Compute the scalogram
    void computeVelocityScalogram()
    {
        // Calculate the scalogram
        mCWT.transform(mVelocitySignal.size(), mVelocitySignal.data());
        auto nScales  = mCWT.getNumberOfScales(); 
        auto nSamples = mCWT.getNumberOfSamples();
#ifndef NDEBUG
        assert(static_cast<int> (mAmplitudeCWT.size()) == nScales*nSamples);
        //assert(static_cast<int> (mCumulativeAmplitudeCWT.size()) ==
        //       nScales*nSamples);
#endif
        auto aPtr = mAmplitudeCWT.data();
        mCWT.getAmplitudeTransform(nSamples, nScales, &aPtr);
        // Accmulate in frequency bins as a function of time
/*
        const double dt2 = 1./(2*mSamplingRate);
        std::fill(mCumulativeAmplitudeCWT.begin(),
                  mCumulativeAmplitudeCWT.end(), 0);
        for (int j = 0; j < nScales; ++j)
        {
            int indx = j*nSamples;
            const double *aScale = mAmplitudeCWT.data() + indx;
            double *cumPtr = mCumulativeAmplitudeCWT.data() + indx;
            cumPtr[0] = (aScale[0] + 0)*dt2;
            for (int i = 1; i < nSamples; ++i)
            {
                cumPtr[i] = cumPtr[i-1] + (aScale[i-1] + aScale[i])*dt2;
            }
        }
*/
        // Dump the scalogram for debugging
/*
        std::string fname{"ampVel.txt"};
        if (mAcceleration){fname = "ampAcc.txt";}
        std::ofstream ofl;
        ofl.open(fname);
        for (int i = 0; i < nSamples; ++i)
        {
            for (int j = 0; j < nScales; ++j)
            {
                ofl << i/mSamplingRate << " " 
                    << mCenterFrequencies[j] << " "
                    << aPtr[j*nSamples + i] << " "
                    << mCumulativeAmplitudeCWT[j*nSamples + i] << std::endl;
            }
            ofl << std::endl;
        }
        ofl.close();
*/
    }
    // Gets the features
    void extractFeatures()
    {
#ifndef NDEBUG
        assert(static_cast<int>(mVelocitySignal.size()) == mTargetSignalLength);
#endif
        auto nScales  = mCWT.getNumberOfScales();
        auto nSamples = mTargetSignalLength;
        auto iPick
            = static_cast<int >(std::round((-mPreArrivalTime - mPickError)
                                           /mTargetSamplingPeriod));
        // Get the temporal features of the noise. 
        // (1) The variance is the signal power minus the DC power.
        auto varianceNoise  = variance(iPick, mVelocitySignal.data());
        // Get difference of min/max amplitude 
        const auto [vMinNoise,  vMaxNoise]
            = std::minmax_element(mVelocitySignal.begin(),
                                  mVelocitySignal.begin() + iPick);
        mTemporalNoiseFeatures.setVariance(varianceNoise);
        mTemporalNoiseFeatures.setMinimumAndMaximumValue(
            std::pair(*vMinNoise, *vMaxNoise));

        // Get the spectral features of the noise.
        SpectralFeatures spectralNoiseFeatures;
        auto dominantFrequencyAmplitude
            = getDominantFrequencyAndAmplitude(nScales, nSamples,
                                               0, iPick,
                                               mCenterFrequencies.data(),
                                               mAmplitudeCWT.data());
        mSpectralNoiseFeatures.setDominantFrequencyAndAmplitude(
            dominantFrequencyAmplitude);

        auto averageFrequencyAmplitude
            = getAverageFrequencyAndAmplitude(nScales, nSamples,
                                              0, iPick,
                                              mCenterFrequencies.data(),
                                              mAmplitudeCWT.data());
        mSpectralNoiseFeatures.setAverageFrequenciesAndAmplitudes(
            averageFrequencyAmplitude);
/*
        auto cumulativeAmplitude
           = getDominantCumulativeAmplitude(nScales, nSamples,
                                            0, iPick,
                                            mCenterFrequencies.data(),
                                            mCumulativeAmplitudeCWT.data());

        spectralNoiseFeatures.setDominantPeriodAndCumulativeAmplitude(
            cumulativeAmplitude[0]);
*/
        // Get the spectral features in windows after arrival
/*
        std::cout << "duration,minNoise,maxNoise,dMinMaxNoise,minSignal,maxSignal,dMinMaxSignal,varianceNoise,varianceSignal,dominantPeriod,amplitudeAtDominantPeriod,dominantCumulativeAmplitudePeriod,cumulativeAmplitdeAtDominantPeriod" << std::endl;
*/
        for (const auto &duration : mDurations)
        {
            // Define the start/end time window indicies
            auto iStart = iPick;
            auto iEnd = static_cast<int>
                        (std::round( (-mPreArrivalTime + duration)
                                     /mTargetSamplingPeriod));
            iEnd = std::min(iEnd, nSamples);
            auto nSubSamples = iEnd - iStart;

            // Get variance in signal
            auto varianceSignal = variance(nSubSamples,
                                           mVelocitySignal.data() + iStart);
            mTemporalSignalFeatures.setVariance(varianceSignal);
            // Get min/max signal
            const auto [vMinSignal, vMaxSignal]
                = std::minmax_element(mVelocitySignal.data() + iStart,
                                      mVelocitySignal.data() + iEnd);
            mTemporalSignalFeatures.setMinimumAndMaximumValue(
                std::pair(*vMinSignal, *vMaxSignal));

            // Extract dominant period
            dominantFrequencyAmplitude
                = getDominantFrequencyAndAmplitude(nScales, nSamples,
                                                   iStart, iEnd,
                                                   mCenterFrequencies.data(),
                                                   mAmplitudeCWT.data());
            mSpectralSignalFeatures.setDominantFrequencyAndAmplitude(
                dominantFrequencyAmplitude); 

            averageFrequencyAmplitude
                = getAverageFrequencyAndAmplitude(nScales, nSamples,
                                                  iStart, iEnd,
                                                  mCenterFrequencies.data(),
                                                  mAmplitudeCWT.data());
            mSpectralSignalFeatures.setAverageFrequenciesAndAmplitudes(
                 averageFrequencyAmplitude);
/*
            cumulativeAmplitude
               = getDominantCumulativeAmplitude(nScales, nSamples,
                                                iStart, iEnd,
                                                mCenterFrequencies.data(),
                                                mCumulativeAmplitudeCWT.data());
            spectralSignalFeatures.setDominantPeriodAndCumulativeAmplitude(
                cumulativeAmplitude[0]);
*/
/*
            std::cout << duration << " "
                      << *vMinNoise  << " " << *vMaxNoise << " " << *vMaxNoise - *vMinNoise << " "
                      << *vMinSignal << " " << *vMaxSignal << " " << *vMaxSignal - *vMinSignal << " " 
                      << varianceNoise << " " << varianceSignal << " " 
                      << dominantPeriodAmplitude.first << " " << dominantPeriodAmplitude.second << " "
                      << cumulativeAmplitude[0].first << " " //<< cumulativeAmplitude[0].second << " "
                      << cumulativeAmplitude[1].first << " " //<< cumulativeAmplitude[1].second << " "
                      << cumulativeAmplitude[2].first << " " //<< cumulativeAmplitude[2].second << " "
                      <<std::endl;
*/
        }
/*
std::cout << "work it:" << std::endl;
std::cout << varianceNoise << " " << varianceSignal << std::endl;
std::cout << *vMinNoise << " " << *vMaxNoise << " " << *vMaxNoise - *vMinNoise << std::endl;
std::cout << *vMinSignal << " " << *vMaxSignal << " " << *vMaxSignal - *vMinSignal << std::endl;
*/
    }
//private:
    // Center frequencies in CWT
    std::vector<double> mCenterFrequencies{1, //1.25, 1.5, 1.75,
                                           2, //2.25, 2.5, 2.75,
                                           3, //3.25, 3.5, 3.75,
                                           4, //4.25, 4.5, 4.75,
                                           5, //5.25, 5.5, 5.75,
                                           6, //6.25, 6.5, 6.75,
                                           7, //7.25, 7.5, 7.75,
                                           8, //8.25, 8.5, 8.75,
                                           9, //9.25, 9.5, 9.75,
                                           10, //10.5,
                                           11, //11.5,
                                           12, //12.5,
                                           13, //13.5,
                                           14, //14.5,
                                           15, 16, 17, 18};
    // Holds signals with units of velocity
    std::vector<double> mVelocitySignal;
    // Workspace
    std::vector<double> mAmplitudeCWT;
    //std::vector<double> mCumulativeAmplitudeCWT; // TODO delete
    std::vector<double> mDurations{1, 2, 3};
    std::string mUnits;
    Preprocess mPreprocess;
    RTSeis::Transforms::Wavelets::Morlet mMorlet;
    RTSeis::Transforms::ContinuousWavelet<double> mCWT;
    Hypocenter mHypocenter;
    Channel mChannel;
    TemporalFeatures mTemporalSignalFeatures;
    TemporalFeatures mTemporalNoiseFeatures;
    SpectralFeatures mSpectralSignalFeatures;
    SpectralFeatures mSpectralNoiseFeatures;
    double mRCFilterQ{0.994};
    double mSamplingRate{100};
    double mSimpleResponse{1}; // Proportional to micrometers
    double mSourceReceiverDistance{-1000};
    double mSourceReceiverBackAzimuth{-1000};
    double mPreArrivalTime{PRE_ARRIVAL_TIME};
    double mPostArrivalTime{POST_ARRIVAL_TIME};
    double mPickError{0.05};
    const double mTargetSamplingRate{TARGET_SAMPLING_RATE};
    const double mTargetSamplingPeriod{TARGET_SAMPLING_PERIOD};
    // The max PGV in the 8.8 Maule event was about. 200 cm/s.
    // This is 200 cm/s which is effectively impossible for UT and
    // Yellowstone and likely indicates a gain problem.
    const double mMaxPeakGroundVelocity{2e6};
    //double mTargetSignalDuration{5}; //TARGET_SIGNAL_DURATION};
    int mTargetSignalLength{500};//TARGET_SIGNAL_LENGTH};
    bool mHaveFeatures{false};
    bool mInitialized{false};
};

/// C'tor
ChannelFeatures::ChannelFeatures(const std::vector<double> &frequencies,
                                 const std::vector<double> &durations,
                                 const double preArrivalTime,
                                 const double postArrivalTime,
                                 const double pickError) :
    pImpl(std::make_unique<FeaturesImpl> (frequencies, durations,
                                          preArrivalTime,
                                          postArrivalTime,
                                          pickError))
{
}

/// Reset class
void ChannelFeatures::clear() noexcept
{
    auto centerFrequencies = pImpl->mCenterFrequencies;
    auto durations = pImpl->mDurations;
    auto [pre, post] = getArrivalTimeProcessingWindow();
    auto pickError = pImpl->mPickError;
    pImpl = std::make_unique<FeaturesImpl> (centerFrequencies, durations,
                                            pre, post, pickError);
}

/// Destructor
ChannelFeatures::~ChannelFeatures() = default;

/// Set the signal
void ChannelFeatures::process(const std::vector<double> &signal,
                              const double arrivalTimeRelativeToStart)
{
    if (signal.empty()){throw std::runtime_error("Signal is empty");}
    process(signal.size(), signal.data(), arrivalTimeRelativeToStart);
}

void ChannelFeatures::process(
    const int n, const double *__restrict__ signal,
    const double arrivalTimeRelativeToStart)
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    // Preprocess signal
    pImpl->mPreprocess.process(n, signal, arrivalTimeRelativeToStart);
    // Check max PGV makes sense
    auto pgvMax = pImpl->mPreprocess.getAbsoluteMaximumPeakGroundVelocity();
    if (pgvMax > pImpl->mMaxPeakGroundVelocity)
    {
        throw std::invalid_argument("Max PGV = "
                 + std::to_string(pgvMax*1.e-4) + " cm/s exceeds "
                 + std::to_string(pImpl->mMaxPeakGroundVelocity*1.e-4)
                 + " cm/s - check response.");
    }
    // Get signal and extract features
    pImpl->mPreprocess.getVelocitySignal(&pImpl->mVelocitySignal);
    // Compute the CWT
    computeVelocityScalogram(pImpl->mVelocitySignal,
                             &pImpl->mAmplitudeCWT,
                             pImpl->mCWT);
    // Extract the time domain and spectral domain features
    pImpl->extractFeatures();
    pImpl->mHaveFeatures = true;
    return;
}

/// Have features?
bool ChannelFeatures::haveFeatures() const noexcept
{
    return pImpl->mHaveFeatures;
}

/// Velocity signal
std::vector<double> ChannelFeatures::getVelocitySignal() const
{
    if (!haveFeatures()){throw std::runtime_error("Signal not yet processed");}
    return pImpl->mVelocitySignal;
}

/// Initialize
void ChannelFeatures::initialize(const Channel &channel)
{
    if (!channel.haveSamplingRate())
    {
        throw std::invalid_argument("Sampling rate not set");
    }
    if (!channel.haveSimpleResponse())
    {
        throw std::invalid_argument("Simple response not set");
    }
    auto samplingRate = channel.getSamplingRate();
    auto simpleResponse = channel.getSimpleResponseValue();
    auto units = channel.getSimpleResponseUnits();
    std::transform(units.begin(), units.end(), units.begin(), ::toupper);
    if (units != "DU/M/S**2" && units != "DU/M/S")
    {
        throw std::runtime_error("units = " + units + " not handled");
    }
    clear();
    pImpl->mUnits = units;
    // Make response proportional to micrometers.  Response units are 
    // 1/meter so to go to 1/micrometer we do 1/(meter*1e6) which effectively
    // divides the input by 1e6.
    pImpl->mChannel = channel;
    pImpl->mSimpleResponse = simpleResponse/1e6; // proportional to micrometers
    pImpl->mSamplingRate = samplingRate;
    pImpl->mInitialized = true;

    // Initialize preprocessor
    pImpl->mPreprocess.initialize(channel.getSamplingRate(),
                                  channel.getSimpleResponseValue(),
                                  channel.getSimpleResponseUnits(),
                                  std::pair(pImpl->mPreArrivalTime,
                                            pImpl->mPostArrivalTime));
    pImpl->mTargetSignalLength = pImpl->mPreprocess.getTargetSignalLength();
    pImpl->mVelocitySignal.resize(pImpl->mTargetSignalLength, 0);
    // Initialize the CWT (need to initialize preprocessor before this)
    pImpl->initializeCWT();
}

bool ChannelFeatures::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

double ChannelFeatures::getSamplingRate() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mSamplingRate;
}

double ChannelFeatures::getSimpleResponseValue() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mSimpleResponse*1e6; // micrometers to meters
}

std::string ChannelFeatures::getSimpleResponseUnits() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mUnits;
}

/// Target information
double ChannelFeatures::getTargetSamplingRate() noexcept
{
    return Preprocess::getTargetSamplingRate();
}

double ChannelFeatures::getTargetSamplingPeriod() noexcept
{
    return Preprocess::getTargetSamplingPeriod();
}

double ChannelFeatures::getTargetSignalDuration() const
{
    return (getTargetSignalLength() - 1)*getTargetSamplingPeriod();
}

int ChannelFeatures::getTargetSignalLength() const 
{
    return pImpl->mPreprocess.getTargetSignalLength();
}

std::pair<double, double>
ChannelFeatures::getArrivalTimeProcessingWindow() const
{
    return std::pair<double, double> {pImpl->mPreArrivalTime,
                                      pImpl->mPostArrivalTime};
}

/// Hypocenter
void ChannelFeatures::setHypocenter(const Hypocenter &hypocenter)
{
    if (!hypocenter.haveLatitude())
    {
        throw std::invalid_argument("Hypocenter latitude not specified");
    }
    if (!hypocenter.haveLongitude())
    {
        throw std::invalid_argument("Hypocenter longitude not specified");
    }
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (pImpl->mChannel.haveLatitude() && pImpl->mChannel.haveLongitude())
    {
        distanceAzimuth(hypocenter, pImpl->mChannel,
                        &pImpl->mSourceReceiverDistance,
                        &pImpl->mSourceReceiverBackAzimuth);
    }
    pImpl->mHypocenter = hypocenter;
}

Hypocenter ChannelFeatures::getHypocenter() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mHypocenter;
}

bool ChannelFeatures::haveHypocenter() const noexcept
{
    return pImpl->mHypocenter.haveLatitude();
}


TemporalFeatures ChannelFeatures::getTemporalNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTemporalNoiseFeatures;
}

TemporalFeatures ChannelFeatures::getTemporalSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTemporalSignalFeatures;
}

SpectralFeatures ChannelFeatures::getSpectralNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mSpectralNoiseFeatures;
}

SpectralFeatures ChannelFeatures::getSpectralSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mSpectralSignalFeatures;
}

double ChannelFeatures::getSourceDepth() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mHypocenter.getDepth();
}

double ChannelFeatures::getSourceReceiverDistance() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mSourceReceiverDistance;
}

double ChannelFeatures::getBackAzimuth() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mSourceReceiverBackAzimuth;
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
/*
template void UUSS::Features::Magnitude::ChannelFeatures::process(
    const std::vector<double> &, double); 
template void UUSS::Features::Magnitude::ChannelFeatures::process(
    const std::vector<float> &, double);
*/
