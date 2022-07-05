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
#include "uuss/features/magnitude/channelFeatures.hpp"
#include "uuss/features/magnitude/channel.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"

using namespace UUSS::Features::Magnitude;

class VerticalChannelFeatures::FeaturesImpl
{
public:
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
template<typename U>
void VerticalChannelFeatures::process(const std::vector<U> &signal,
                                      const double arrivalTimeRelativeToStart)
{
    if (signal.empty()){throw std::runtime_error("Signal is empty");}
    process(signal.size(), signal.data(), arrivalTimeRelativeToStart);
}

template<typename U>
void VerticalChannelFeatures::process(
    const int n, const U *__restrict__ signal,
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

double VerticalChannelFeatures::getTargetSignalDuration() noexcept
{
    return ChannelFeatures::getTargetSignalDuration();
}

int VerticalChannelFeatures::getTargetSignalLength() noexcept
{
    return ChannelFeatures::getTargetSignalLength();
}

std::pair<double, double>
VerticalChannelFeatures::getArrivalTimeProcessingWindow() noexcept
{
    return ChannelFeatures::getArrivalTimeProcessingWindow();
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

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template void UUSS::Features::Magnitude::VerticalChannelFeatures::process(
    const std::vector<double> &, double); 
template void UUSS::Features::Magnitude::VerticalChannelFeatures::process(
    const std::vector<float> &, double);
