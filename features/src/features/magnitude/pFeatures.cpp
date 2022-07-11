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
#include "uuss/features/magnitude/pFeatures.hpp"
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
#define P_PICK_ERROR 0.05           // Alysha's P pickers are usually within 5 samples
/*
#define TARGET_SIGNAL_LENGTH 500    // 1s before to 4s after
#define PRE_ARRIVAL_TIME 1          // 1s before P arrival
#define POST_ARRIVAL_TIME 4         // 4s after P arrival
#define P_PICK_ERROR 0.05           // Alysha's P pickers are usually within 5 samples
#define TARGET_SIGNAL_DURATION (TARGET_SIGNAL_LENGTH - 1)*TARGET_SAMPLING_PERIOD
*/

using namespace UUSS::Features::Magnitude;

class PFeatures::FeaturesImpl
{
public:
    FeaturesImpl( //const std::vector<double> &frequencies,
                  //const std::vector<double> &durations,
                  //const double preArrivalTime  = PRE_ARRIVAL_TIME,
                  //const double postArrivalTime = POST_ARRIVAL_TIME,
                  //const double pickError       = P_PICK_ERROR
                )
/*
        mCenterFrequencies(frequencies),
        mDurations(durations),
        mPreArrivalTime(preArrivalTime),
        mPostArrivalTime(postArrivalTime),
        mPickError(pickError)
*/
    {
        if (mPreArrivalTime > 0)
        {
            throw std::invalid_argument("Pre arrival time must negative");
        }
        if (mPostArrivalTime < 0)
        {
            throw std::invalid_argument("Post arrival time must be positive");
        }
        if (mPickError < 0)
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
            scales[i] = (omega0*Preprocess::getTargetSamplingRate())
                       /(2*M_PI*mCenterFrequencies[i]);
        }
        mCWT.initialize(nSamples,
                        scales.size(), scales.data(),
                        mMorlet,
                        Preprocess::getTargetSamplingRate());
        mAmplitudeCWT.resize(scales.size()*nSamples, 0);
    }
    // Gets the features
    void extractFeatures()
    {
        auto targetDT = Preprocess::getTargetSamplingPeriod(); 
#ifndef NDEBUG
        assert(static_cast<int>(mVelocitySignal.size()) ==
               mCWT.getNumberOfSamples());
#endif
        auto nScales  = mCWT.getNumberOfScales();
        auto nSamples = static_cast<int> (mVelocitySignal.size());
        auto iPick = static_cast<int>
                     (std::round((-mPreArrivalTime - mPickError)/targetDT));
        mTemporalNoiseFeatures
             = getTimeDomainFeatures(0, iPick, mVelocitySignal);

        mSpectralNoiseFeatures
            = getSpectralDomainFeatures(nSamples, nScales, 
                                        0, iPick,
                                        mCenterFrequencies,
                                        mAmplitudeCWT);
        // Get the spectral features in windows after arrival
        for (const auto &duration : mDurations)
        {
            // Define the start/end time window indicies
            auto iStart = iPick;
            auto iEnd = static_cast<int>
                        (std::round( (-mPreArrivalTime + duration)/targetDT));
            iEnd = std::min(iEnd, nSamples);

            mTemporalSignalFeatures
                = getTimeDomainFeatures(iStart, iEnd, mVelocitySignal);

            mSpectralSignalFeatures
               = getSpectralDomainFeatures(nSamples, nScales, 
                                           iStart, iEnd,
                                           mCenterFrequencies,
                                           mAmplitudeCWT);
        }
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
    std::vector<double> mDurations{2.5};
    Preprocess mPreprocess;
    RTSeis::Transforms::Wavelets::Morlet mMorlet;
    RTSeis::Transforms::ContinuousWavelet<double> mCWT;
    Hypocenter mHypocenter;
    Channel mChannel;
    TemporalFeatures mTemporalSignalFeatures;
    TemporalFeatures mTemporalNoiseFeatures;
    SpectralFeatures mSpectralSignalFeatures;
    SpectralFeatures mSpectralNoiseFeatures;
    double mSourceReceiverDistance{-1000};
    double mSourceReceiverBackAzimuth{-1000};
    double mPreArrivalTime{PRE_ARRIVAL_TIME};
    double mPostArrivalTime{POST_ARRIVAL_TIME};
    double mPickError{P_PICK_ERROR};
    // The max PGV in the 8.8 Maule event was about. 200 cm/s.
    // This is 200 cm/s which is effectively impossible for UT and
    // Yellowstone and likely indicates a gain problem.
    const double mMaxPeakGroundVelocity{2e6};
    bool mHaveFeatures{false};
    bool mInitialized{false};
};

/// C'tor
PFeatures::PFeatures() :
    pImpl(std::make_unique<FeaturesImpl> ())
{
}

/// Reset class
void PFeatures::clear() noexcept
{
    pImpl = std::make_unique<FeaturesImpl> ();
}

/// Destructor
PFeatures::~PFeatures() = default;

/// Set the signal
void PFeatures::process(const std::vector<double> &signal,
                              const double arrivalTimeRelativeToStart)
{
    if (signal.empty()){throw std::runtime_error("Signal is empty");}
    process(signal.size(), signal.data(), arrivalTimeRelativeToStart);
}

void PFeatures::process(
    const int n, const double *__restrict__ signal,
    const double arrivalTimeRelativeToStart)
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (signal == nullptr){throw std::invalid_argument("signal is NULL");}
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
}

/// Have features?
bool PFeatures::haveFeatures() const noexcept
{
    return pImpl->mHaveFeatures;
}

/// Velocity signal
std::vector<double> PFeatures::getVelocitySignal() const
{
    if (!haveFeatures()){throw std::runtime_error("Signal not yet processed");}
    return pImpl->mVelocitySignal;
}

/// Initialize
void PFeatures::initialize(const Channel &channel)
{
    if (!channel.haveSamplingRate())
    {
        throw std::invalid_argument("Sampling rate not set");
    }
    if (!channel.haveSimpleResponse())
    {
        throw std::invalid_argument("Simple response not set");
    }
    auto units = channel.getSimpleResponseUnits();
    std::transform(units.begin(), units.end(), units.begin(), ::toupper);
    if (units != "DU/M/S**2" && units != "DU/M/S")
    {
        throw std::invalid_argument("units = " + units + " not handled");
    }
    clear();
    pImpl->mChannel = channel;
    // Initialize preprocessor
    pImpl->mPreprocess.initialize(channel.getSamplingRate(),
                                  channel.getSimpleResponseValue(),
                                  channel.getSimpleResponseUnits(),
                                  std::pair(pImpl->mPreArrivalTime,
                                            pImpl->mPostArrivalTime));
    auto targetSignalLength = pImpl->mPreprocess.getTargetSignalLength();
    pImpl->mVelocitySignal.resize(targetSignalLength, 0);
    // Initialize the CWT (need to initialize preprocessor before this)
    pImpl->initializeCWT();
    pImpl->mInitialized = true;
}

bool PFeatures::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

double PFeatures::getSamplingRate() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mPreprocess.getSamplingRate();
}

double PFeatures::getSimpleResponseValue() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mChannel.getSimpleResponseValue();
}

std::string PFeatures::getSimpleResponseUnits() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mChannel.getSimpleResponseUnits();
}

/// Target information
double PFeatures::getTargetSamplingRate() noexcept
{
    return Preprocess::getTargetSamplingRate();
}

double PFeatures::getTargetSamplingPeriod() noexcept
{
    return Preprocess::getTargetSamplingPeriod();
}

double PFeatures::getTargetSignalDuration() const
{
    return (getTargetSignalLength() - 1)*getTargetSamplingPeriod();
}

int PFeatures::getTargetSignalLength() const 
{
    return pImpl->mPreprocess.getTargetSignalLength();
}

std::pair<double, double>
PFeatures::getArrivalTimeProcessingWindow() const
{
    return std::pair<double, double> {pImpl->mPreArrivalTime,
                                      pImpl->mPostArrivalTime};
}

/// Hypocenter
void PFeatures::setHypocenter(const Hypocenter &hypocenter)
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

Hypocenter PFeatures::getHypocenter() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mHypocenter;
}

bool PFeatures::haveHypocenter() const noexcept
{
    return pImpl->mHypocenter.haveLatitude();
}


TemporalFeatures PFeatures::getTemporalNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTemporalNoiseFeatures;
}

TemporalFeatures PFeatures::getTemporalSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTemporalSignalFeatures;
}

SpectralFeatures PFeatures::getSpectralNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mSpectralNoiseFeatures;
}

SpectralFeatures PFeatures::getSpectralSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mSpectralSignalFeatures;
}

double PFeatures::getSourceDepth() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mHypocenter.getDepth();
}

double PFeatures::getSourceReceiverDistance() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mSourceReceiverDistance;
}

double PFeatures::getBackAzimuth() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mSourceReceiverBackAzimuth;
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
/*
template void UUSS::Features::Magnitude::PFeatures::process(
    const std::vector<double> &, double); 
template void UUSS::Features::Magnitude::PFeatures::process(
    const std::vector<float> &, double);
*/
