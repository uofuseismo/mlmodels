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
#include "uuss/features/magnitude/verticalChannelFeatures.hpp"
#include "uuss/features/magnitude/temporalFeatures.hpp"
#include "uuss/features/magnitude/spectralFeatures.hpp"
#include "uuss/features/magnitude/channelFeatures.hpp"
#include "uuss/features/magnitude/channel.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100
#define TARGET_SIGNAL_LENGTH 500    // 1s before to 4s after
#define PRE_ARRIVAL_TIME 1          // 1s before P arrival
#define POST_ARRIVAL_TIME 4         // 4s after P arrival
#define P_PICK_ERROR 0.05           // Alysha's P pickers are usually within 5 samples

using namespace UUSS::Features::Magnitude;

class VerticalChannelFeatures::FeaturesImpl
{
public:
    FeaturesImpl() :
        mChannelFeatures(mFrequencies, mDurations,
                         -PRE_ARRIVAL_TIME,
                         POST_ARRIVAL_TIME,
                         P_PICK_ERROR)
    {
    }
    const std::vector<double> mFrequencies{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::vector<double> mDurations{2.5}; // Saturate at about M5
    ChannelFeatures mChannelFeatures;
};

/// C'tor
VerticalChannelFeatures::VerticalChannelFeatures() :
    pImpl(std::make_unique<FeaturesImpl> ())
{
}

/// Reset class
void VerticalChannelFeatures::clear() noexcept
{
    pImpl = std::make_unique<FeaturesImpl> ();
}

/// Destructor
VerticalChannelFeatures::~VerticalChannelFeatures() = default;

/// Set the signal
void VerticalChannelFeatures::process(const std::vector<double> &signal,
                                      const double arrivalTimeRelativeToStart)
{
    if (signal.empty()){throw std::runtime_error("Signal is empty");}
    process(signal.size(), signal.data(), arrivalTimeRelativeToStart);
}

void VerticalChannelFeatures::process(
    const int n, const double *__restrict__ signal,
    const double arrivalTimeRelativeToStart)
{
    pImpl->mChannelFeatures.process(n, signal, arrivalTimeRelativeToStart);
}

/// Have signal?
bool VerticalChannelFeatures::haveSignal() const noexcept
{
    return pImpl->mChannelFeatures.haveFeatures();
}

/// Velocity signal
std::vector<double> VerticalChannelFeatures::getVelocitySignal() const
{
    return pImpl->mChannelFeatures.getVelocitySignal();
}

/// Initialize
void VerticalChannelFeatures::initialize(const Channel &channel)
{
    auto channelCode = channel.getChannelCode();
    if (channelCode.back() != 'Z')
    {
        throw std::invalid_argument("Invalid channel code");
    }
    pImpl->mChannelFeatures.initialize(channel);
}

bool VerticalChannelFeatures::isInitialized() const noexcept
{
    return pImpl->mChannelFeatures.isInitialized();
}

double VerticalChannelFeatures::getSamplingRate() const
{
    return pImpl->mChannelFeatures.getSamplingRate();
}

double VerticalChannelFeatures::getSimpleResponseValue() const
{
    return pImpl->mChannelFeatures.getSimpleResponseValue();
}

std::string VerticalChannelFeatures::getSimpleResponseUnits() const
{
    return pImpl->mChannelFeatures.getSimpleResponseUnits();
}

/// Target information
double VerticalChannelFeatures::getTargetSamplingRate() noexcept
{
    return ChannelFeatures::getTargetSamplingRate();
}

double VerticalChannelFeatures::getTargetSamplingPeriod() noexcept
{
    return ChannelFeatures::getTargetSamplingPeriod();
}

double VerticalChannelFeatures::getTargetSignalDuration() const noexcept
{
    return pImpl->mChannelFeatures.getTargetSignalDuration();
}

int VerticalChannelFeatures::getTargetSignalLength() const noexcept
{
    return pImpl->mChannelFeatures.getTargetSignalLength();
}

std::pair<double, double>
VerticalChannelFeatures::getArrivalTimeProcessingWindow() const noexcept
{
    return pImpl->mChannelFeatures.getArrivalTimeProcessingWindow();
}

/// Hypocenter
void VerticalChannelFeatures::setHypocenter(const Hypocenter &hypocenter)
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
    pImpl->mChannelFeatures.setHypocenter(hypocenter);
}

Hypocenter VerticalChannelFeatures::getHypocenter() const
{
    return pImpl->mChannelFeatures.getHypocenter();
}

bool VerticalChannelFeatures::haveHypocenter() const noexcept
{
    return pImpl->mChannelFeatures.haveHypocenter();
}

TemporalFeatures VerticalChannelFeatures::getTemporalNoiseFeatures() const
{
    return pImpl->mChannelFeatures.getTemporalNoiseFeatures();
}

TemporalFeatures VerticalChannelFeatures::getTemporalSignalFeatures() const
{
    return pImpl->mChannelFeatures.getTemporalSignalFeatures();
}

SpectralFeatures VerticalChannelFeatures::getSpectralNoiseFeatures() const
{
    return pImpl->mChannelFeatures.getSpectralNoiseFeatures();
}

SpectralFeatures VerticalChannelFeatures::getSpectralSignalFeatures() const
{
    return pImpl->mChannelFeatures.getSpectralSignalFeatures();
}

double VerticalChannelFeatures::getSourceDepth() const
{
    return pImpl->mChannelFeatures.getSourceDepth();
}

double VerticalChannelFeatures::getSourceReceiverDistance() const
{
    return pImpl->mChannelFeatures.getSourceReceiverDistance();
}

double VerticalChannelFeatures::getBackAzimuth() const
{
    return pImpl->mChannelFeatures.getBackAzimuth();
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
/*
template void UUSS::Features::Magnitude::VerticalChannelFeatures::process(
    const std::vector<double> &, double); 
template void UUSS::Features::Magnitude::VerticalChannelFeatures::process(
    const std::vector<float> &, double);
*/
