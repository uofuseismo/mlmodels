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
#include "uuss/features/magnitude/sFeatures.hpp"
#include "uuss/features/magnitude/temporalFeatures.hpp"
#include "uuss/features/magnitude/spectralFeatures.hpp"
#include "uuss/features/magnitude/channel.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"
#include "uuss/features/magnitude/preprocess.hpp"
#include "private/magnitudeUtilities.hpp"

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100
#define TARGET_SIGNAL_LENGTH 650    // 1.5s before to 5s after
#define PRE_ARRIVAL_TIME -1.5       // 1.5s before S arrival
#define POST_ARRIVAL_TIME 5         // 5s after S arrival
#define S_PICK_ERROR 0.10           // Alysha's S pickers are about twice as noise as the P pick so 0.05 seconds -> 0.1 seconds

using namespace UUSS::Features::Magnitude;

class SFeatures::FeaturesImpl
{
public:
    FeaturesImpl(const std::vector<double> &frequencies = std::vector<double> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
                 const std::vector<double> &durations = std::vector<double> {3},
                 const double preArrivalTime  = PRE_ARRIVAL_TIME,
                 const double postArrivalTime = POST_ARRIVAL_TIME,
                 const double pickError       = S_PICK_ERROR) :
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
    }
    // Initialize the CWT
    void initializeCWT()
    {
        auto nSamples = mNorthPreprocess.getTargetSignalLength();
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
    const std::vector<double> mCenterFrequencies{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::vector<double> mDurations{4}; // No idea on saturation - probably bigger than M5 

    std::vector<double> mAmplitudeCWT;
    std::vector<double> mRadialVelocity;
    std::vector<double> mTransverseVelocity;

    RTSeis::Transforms::Wavelets::Morlet mMorlet;
    RTSeis::Transforms::ContinuousWavelet<double> mCWT;
    Hypocenter mHypocenter;
    Preprocess mNorthPreprocess;
    Preprocess mEastPreprocess;
    Channel mNorthChannel;
    Channel mEastChannel;
    double mSourceReceiverDistance{-1000};
    double mSourceReceiverBackAzimuth{-1000};
    double mPreArrivalTime{PRE_ARRIVAL_TIME};
    double mPostArrivalTime{POST_ARRIVAL_TIME};
    double mPickError{S_PICK_ERROR};
    bool mInitialized{false};
};

/// C'tor
SFeatures::SFeatures() :
    pImpl(std::make_unique<FeaturesImpl> ())
{
}

/// Destructor
SFeatures::~SFeatures() = default;

/// Initialized?
bool SFeatures::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

/// Process the signal
void SFeatures::process(const std::vector<double> &nSignal,
                        const std::vector<double> &eSignal,
                        const double arrivalTimeRelativeToStart)
{
    if (nSignal.size() != eSignal.size())
    {
        throw std::invalid_argument("Inconsistent signal lengths");
    }
    process(nSignal.size(),
            nSignal.data(), eSignal.data(),
            arrivalTimeRelativeToStart);
}

void SFeatures::process(
    const int n,
    const double *__restrict__ nSignal,
    const double *__restrict__ eSignal,
    const double arrivalTimeRelativeToStart)
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    if (nSignal == nullptr){throw std::invalid_argument("nSignal is NULL");}
    if (eSignal == nullptr){throw std::invalid_argument("eSignal is NULL");}
    // Preprocess the signals
    pImpl->mNorthPreprocess.process(n, nSignal, arrivalTimeRelativeToStart);
    pImpl->mEastPreprocess.process(n, eSignal, arrivalTimeRelativeToStart);
    // Get the pre-processed signals
/*
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
*/
}

/// Hypocenter 
void SFeatures::setHypocenter(const Hypocenter &hypocenter)
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
    if (pImpl->mNorthChannel.haveLatitude() &&
        pImpl->mNorthChannel.haveLongitude())
    {
        distanceAzimuth(hypocenter, pImpl->mNorthChannel,
                        &pImpl->mSourceReceiverDistance,
                        &pImpl->mSourceReceiverBackAzimuth);
    }
    pImpl->mHypocenter = hypocenter;
}

double SFeatures::getBackAzimuth() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mSourceReceiverBackAzimuth;
}

double SFeatures::getSourceReceiverDistance() const
{
    if (!haveHypocenter()){throw std::runtime_error("Hypocenter not set");}
    return pImpl->mSourceReceiverDistance;
}

bool SFeatures::haveHypocenter() const noexcept
{
    return pImpl->mHypocenter.haveLatitude();
}

/// Target information
double SFeatures::getTargetSamplingRate() noexcept
{
    return Preprocess::getTargetSamplingRate();
}

double SFeatures::getTargetSamplingPeriod() noexcept
{
    return Preprocess::getTargetSamplingPeriod();
}

double SFeatures::getTargetSignalDuration() const
{
    return (getTargetSignalLength() - 1)*getTargetSamplingPeriod();
}

int SFeatures::getTargetSignalLength() const
{
    return pImpl->mNorthPreprocess.getTargetSignalLength();
}
