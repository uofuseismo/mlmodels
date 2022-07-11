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
#include <rtseis/rotate/utilities.hpp>
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
/*
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
*/
    // Gets the features
    void extractFeatures()
    {
        auto targetDT = Preprocess::getTargetSamplingPeriod(); 
#ifndef NDEBUG
        assert(static_cast<int>(mRadialVelocity.size()) ==
               mCWT.getNumberOfSamples());
        assert(static_cast<int>(mTransverseVelocity.size()) ==
               mCWT.getNumberOfSamples());
#endif
        auto nScales  = mCWT.getNumberOfScales();
        auto nSamples = mCWT.getNumberOfSamples();
        auto iPick = static_cast<int>
                     (std::round((-mPreArrivalTime - mPickError)/targetDT));
        mRadialTemporalNoiseFeatures
             = getTimeDomainFeatures(0, iPick, mRadialVelocity);
        mTransverseTemporalNoiseFeatures
             = getTimeDomainFeatures(0, iPick, mTransverseVelocity);

        mRadialSpectralNoiseFeatures
            = getSpectralDomainFeatures(nSamples, nScales, 
                                        0, iPick,
                                        mCenterFrequencies,
                                        mRadialAmplitudeCWT);
        mTransverseSpectralNoiseFeatures
            = getSpectralDomainFeatures(nSamples, nScales, 
                                        0, iPick,
                                        mCenterFrequencies,
                                        mTransverseAmplitudeCWT);
        // Get the spectral features in windows after arrival
        for (const auto &duration : mDurations)
        {
            // Define the start/end time window indicies
            auto iStart = iPick;
            auto iEnd = static_cast<int>
                        (std::round( (-mPreArrivalTime + duration)/targetDT));
            iEnd = std::min(iEnd, nSamples);

            mRadialTemporalSignalFeatures
                = getTimeDomainFeatures(iStart, iEnd, mRadialVelocity);
            mTransverseTemporalSignalFeatures
                = getTimeDomainFeatures(iStart, iEnd, mTransverseVelocity);

            mRadialSpectralSignalFeatures
               = getSpectralDomainFeatures(nSamples, nScales, 
                                           iStart, iEnd,
                                           mCenterFrequencies,
                                           mRadialAmplitudeCWT);
            mTransverseSpectralSignalFeatures
               = getSpectralDomainFeatures(nSamples, nScales, 
                                           iStart, iEnd,
                                           mCenterFrequencies,
                                           mTransverseAmplitudeCWT);
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
        mRadialAmplitudeCWT.resize(scales.size()*nSamples, 0);
        mTransverseAmplitudeCWT.resize(scales.size()*nSamples, 0);
    }
    const std::vector<double> mCenterFrequencies{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::vector<double> mDurations{4}; // No idea on saturation - probably bigger than M5 

    std::vector<double> mRadialAmplitudeCWT;
    std::vector<double> mTransverseAmplitudeCWT;
    std::vector<double> mRadialVelocity;
    std::vector<double> mTransverseVelocity;
    std::vector<double> mNorthVelocity;
    std::vector<double> mEastVelocity;

    RTSeis::Transforms::Wavelets::Morlet mMorlet;
    RTSeis::Transforms::ContinuousWavelet<double> mCWT;
    Hypocenter mHypocenter;
    Preprocess mNorthPreprocess;
    Preprocess mEastPreprocess;
    Channel mNorthChannel;
    Channel mEastChannel;
    TemporalFeatures mRadialTemporalSignalFeatures;
    TemporalFeatures mRadialTemporalNoiseFeatures;
    SpectralFeatures mRadialSpectralSignalFeatures;
    SpectralFeatures mRadialSpectralNoiseFeatures;
    TemporalFeatures mTransverseTemporalSignalFeatures;
    TemporalFeatures mTransverseTemporalNoiseFeatures;
    SpectralFeatures mTransverseSpectralSignalFeatures;
    SpectralFeatures mTransverseSpectralNoiseFeatures;
    double mSourceReceiverDistance{-1000};
    double mSourceReceiverBackAzimuth{-1000};
    double mPreArrivalTime{PRE_ARRIVAL_TIME};
    double mPostArrivalTime{POST_ARRIVAL_TIME};
    double mPickError{S_PICK_ERROR};
    // The max PGV in the 8.8 Maule event was about. 200 cm/s.
    // This is 200 cm/s which is effectively impossible for UT and
    // Yellowstone and likely indicates a gain problem.
    const double mMaxPeakGroundVelocity{2e6};
    // Sometimes non-vertical channels tilt out of alignment and one channel's
    // amplitude is much bigger than the other channel's amplitude.
    const double mMaxPGVRatio{100};
    bool mInitialized{false};
    bool mHaveFeatures{false};
};

/// C'tor
SFeatures::SFeatures() :
    pImpl(std::make_unique<FeaturesImpl> ())
{
}

/// Reset class
void SFeatures::clear() noexcept
{
    pImpl = std::make_unique<FeaturesImpl> ();
}

/// Destructor
SFeatures::~SFeatures() = default;

/// Initialized?
bool SFeatures::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

/// Initialize
void SFeatures::initialize(const Channel &north, const Channel &east)
{
    if (!north.haveSamplingRate())
    {
        throw std::invalid_argument("North sampling rate not set");
    }
    if (!east.haveSamplingRate())
    {
        throw std::invalid_argument("East sampling rate not set");
    }
    if (!north.haveSimpleResponse())
    {
        throw std::invalid_argument("North simple response not set");
    }
    if (!east.haveSimpleResponse())
    {
        throw std::invalid_argument("East simple response not set");
    }
    auto nUnits = north.getSimpleResponseUnits();
    std::transform(nUnits.begin(), nUnits.end(), nUnits.begin(), ::toupper);
    if (nUnits != "DU/M/S**2" && nUnits != "DU/M/S")
    {
        throw std::runtime_error("north units = " + nUnits + " not handled");
    }
    auto eUnits = east.getSimpleResponseUnits();
    std::transform(eUnits.begin(), eUnits.end(), eUnits.begin(), ::toupper);
    if (eUnits != "DU/M/S**2" && eUnits != "DU/M/S")
    {
        throw std::runtime_error("east units = " + eUnits + " not handled");
    }
    auto northAzimuth = north.getAzimuth();
    auto eastAzimuth = east.getAzimuth();
    if (std::abs(northAzimuth - eastAzimuth) > 1.e-4)
    {
        throw std::invalid_argument(
            "Component azimuths must be 90 degrees apart");
    }
    if (!north.haveLatitude() || !north.haveLongitude())
    {
        throw std::invalid_argument("lat/lon not set on north channel");
    }
    if (!east.haveLatitude() || !east.haveLongitude())
    {
        throw std::invalid_argument("lat/lon not set on east channel");
    }
    if (std::abs(north.getLatitude() - east.getLatitude()) > 1.e-4)
    {
        throw std::invalid_argument("latitude inconsistent");
    }
    if (std::abs(north.getLongitude() - north.getLongitude()) > 1.e-4)
    {
        throw std::invalid_argument("longitude inconsistent");
    }
    // Initialize
    clear();
    pImpl->mNorthChannel = north;
    pImpl->mEastChannel = east;
    // Initialize preprocessor
    pImpl->mNorthPreprocess.initialize(north.getSamplingRate(),
                                       north.getSimpleResponseValue(),
                                       north.getSimpleResponseUnits(),
                                       std::pair(pImpl->mPreArrivalTime,
                                                 pImpl->mPostArrivalTime));
    pImpl->mEastPreprocess.initialize(east.getSamplingRate(),
                                      east.getSimpleResponseValue(),
                                      east.getSimpleResponseUnits(),
                                      std::pair(pImpl->mPreArrivalTime,
                                                pImpl->mPostArrivalTime));
    auto targetSignalLength = pImpl->mNorthPreprocess.getTargetSignalLength();
#ifndef NDEBUG
    assert(targetSignalLength ==
           pImpl->mEastPreprocess.getTargetSignalLength());
#endif
    pImpl->mNorthVelocity.resize(targetSignalLength, 0);
    pImpl->mEastVelocity.resize(targetSignalLength, 0); 
    pImpl->mRadialVelocity.resize(targetSignalLength, 0);
    pImpl->mTransverseVelocity.resize(targetSignalLength, 0);
    // Initialize the CWT (need to initialize preprocessor before this)
    pImpl->initializeCWT();
    pImpl->mInitialized = true;
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
    // Check max PGV makes sense
    auto pgvMaxN
        = pImpl->mNorthPreprocess.getAbsoluteMaximumPeakGroundVelocity();
    auto pgvMaxE
        = pImpl->mEastPreprocess.getAbsoluteMaximumPeakGroundVelocity();
    if (pgvMaxN > pImpl->mMaxPeakGroundVelocity)
    {
        throw std::invalid_argument("Max PGV = "
                 + std::to_string(pgvMaxN*1.e-4)
                 + " cm/s on north channel exceeds "
                 + std::to_string(pImpl->mMaxPeakGroundVelocity*1.e-4)
                 + " cm/s - check north response.");
    }
    if (pgvMaxE > pImpl->mMaxPeakGroundVelocity)
    {
        throw std::invalid_argument("Max PGV = "
                 + std::to_string(pgvMaxE*1.e-4)
                 + " cm/s on east channel exceeds "
                 + std::to_string(pImpl->mMaxPeakGroundVelocity*1.e-4)
                 + " cm/s - check east response.");
    }
    if (pgvMaxN > pgvMaxE*pImpl->mMaxPGVRatio)
    {
        throw std::invalid_argument("Max PGV on north > 100x bigger than east");
    }
    if (pgvMaxE > pgvMaxN*pImpl->mMaxPGVRatio)
    {
        throw std::invalid_argument("Max PGV on east > 100x bigger than north");
    }
    // Get signal and extract features
    pImpl->mNorthPreprocess.getVelocitySignal(&pImpl->mNorthVelocity);
    pImpl->mEastPreprocess.getVelocitySignal(&pImpl->mEastVelocity);
#ifndef NDEBUG
    assert(pImpl->mNorthVelocity.size() == pImpl->mEastVelocity.size());
#endif
    auto nSamples = getTargetSignalLength();
    auto backAzimuth = getBackAzimuth();
    if (pImpl->mRadialVelocity.size() != pImpl->mNorthVelocity.size())
    {
        pImpl->mRadialVelocity.resize(pImpl->mNorthVelocity.size(), 0);
    }
    if (pImpl->mTransverseVelocity.size() != pImpl->mEastVelocity.size())
    {
        pImpl->mTransverseVelocity.resize(pImpl->mEastVelocity.size(), 0);
    }
    // Handle rotation of components
    auto northAzimuth = pImpl->mNorthChannel.getAzimuth();
    backAzimuth = backAzimuth - northAzimuth;
    if (backAzimuth < 0){backAzimuth = backAzimuth + 360;}
    // Possibly flip polarity of other channel
    auto eastAzimuth = pImpl->mEastChannel.getAzimuth();
    bool lFlipTransverse = false;
    if (std::abs(northAzimuth + 90 - eastAzimuth) > 1.e-4)
    {
        std::cout << "Flipping rotation" << std::endl;
        lFlipTransverse = true;
    }

    auto radialPtr = pImpl->mRadialVelocity.data();
    auto transversePtr = pImpl->mTransverseVelocity.data();
    RTSeis::Rotate::northEastToRadialTransverse(nSamples,
                                                backAzimuth,
                                                pImpl->mNorthVelocity.data(),
                                                pImpl->mEastVelocity.data(),
                                                &radialPtr, &transversePtr);
    if (lFlipTransverse)
    {
        std::transform(pImpl->mTransverseVelocity.begin(),
                       pImpl->mTransverseVelocity.end(),
                       pImpl->mTransverseVelocity.begin(),
                       [](double t)
                       {
                           return -t;
                       }); 
    }
    // Compute the CWTs
    computeVelocityScalogram(pImpl->mRadialVelocity,
                             &pImpl->mRadialAmplitudeCWT,
                             pImpl->mCWT);
    computeVelocityScalogram(pImpl->mTransverseVelocity,
                             &pImpl->mTransverseAmplitudeCWT,
                             pImpl->mCWT);
    // Extract the time domain and spectral domain features
    pImpl->extractFeatures();
    pImpl->mHaveFeatures = true;
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

/// Have features?
bool SFeatures::haveFeatures() const noexcept
{
    return pImpl->mHaveFeatures;
}

/// Temporal features
TemporalFeatures SFeatures::getRadialTemporalNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mRadialTemporalNoiseFeatures;
}

TemporalFeatures SFeatures::getRadialTemporalSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mRadialTemporalSignalFeatures;
}

TemporalFeatures SFeatures::getTransverseTemporalNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTransverseTemporalNoiseFeatures;
}

TemporalFeatures SFeatures::getTransverseTemporalSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTransverseTemporalSignalFeatures;
}

/// Spectral features
SpectralFeatures SFeatures::getRadialSpectralNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mRadialSpectralNoiseFeatures;
}

SpectralFeatures SFeatures::getRadialSpectralSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mRadialSpectralSignalFeatures;
}

SpectralFeatures SFeatures::getTransverseSpectralNoiseFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTransverseSpectralNoiseFeatures;
}

SpectralFeatures SFeatures::getTransverseSpectralSignalFeatures() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTransverseSpectralSignalFeatures;
}

/// Radial velocity signal
std::vector<double> SFeatures::getRadialVelocitySignal() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mRadialVelocity;
}

/// Transverse velocity signal
std::vector<double> SFeatures::getTransverseVelocitySignal() const
{
    if (!haveFeatures()){throw std::runtime_error("Features not computed");}
    return pImpl->mTransverseVelocity;
}
