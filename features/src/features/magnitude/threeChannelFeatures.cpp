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
#include "uuss/features/magnitude/threeChannelFeatures.hpp"
#include "uuss/features/magnitude/temporalFeatures.hpp"
#include "uuss/features/magnitude/spectralFeatures.hpp"
#include "uuss/features/magnitude/channelFeatures.hpp"
#include "uuss/features/magnitude/channel.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100
#define TARGET_SIGNAL_LENGTH 650    // 1.5s before to 5s after
#define PRE_ARRIVAL_TIME 1.5        // 1.5s before S arrival
#define POST_ARRIVAL_TIME 5         // 5s after S arrival
#define S_PICK_ERROR 0.10           // Alysha's S pickers are about twice as noise as the P pick so 0.05 seconds -> 0.1 seconds

using namespace UUSS::Features::Magnitude;

class ThreeChannelFeatures::FeaturesImpl
{
public:
    FeaturesImpl() :
        mChannelFeatures(mFrequencies, mDurations,
                        -PRE_ARRIVAL_TIME,
                         POST_ARRIVAL_TIME,
                         S_PICK_ERROR)
    {
    }
    const std::vector<double> mFrequencies{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::vector<double> mDurations{4}; // No idea on saturation - probably bigger than M5 
    std::vector<double> mVerticalVelocity;
    std::vector<double> mRadialVelocity;
    std::vector<double> mTransverseVelociy;
    ChannelFeatures mChannelFeatures;
    double mBackAzimuth{0};
};

/// C'tor
ThreeChannelFeatures::ThreeChannelFeatures() :
    pImpl(std::make_unique<FeaturesImpl> ())
{
}

/// Destructor
ThreeChannelFeatures::~ThreeChannelFeatures() = default;

/// Initialized?
bool ThreeChannelFeatures::isInitialized() const noexcept
{
    return pImpl->mChannelFeatures.isInitialized();
}

/// Hypocenter 
void ThreeChannelFeatures::setHypocenter(const Hypocenter &hypo)
{
    pImpl->mChannelFeatures.setHypocenter(hypo);
}

double ThreeChannelFeatures::getBackAzimuth() const
{
    return pImpl->mChannelFeatures.getBackAzimuth();
}

double ThreeChannelFeatures::getSourceReceiverDistance() const
{
    return pImpl->mChannelFeatures.getBackAzimuth();
}

bool ThreeChannelFeatures::haveHypocenter() const noexcept
{
    return pImpl->mChannelFeatures.haveHypocenter();
}

/// Target information
double ThreeChannelFeatures::getTargetSamplingRate() noexcept
{
    return ChannelFeatures::getTargetSamplingRate();
}

double ThreeChannelFeatures::getTargetSamplingPeriod() noexcept
{
    return ChannelFeatures::getTargetSamplingPeriod();
}

double ThreeChannelFeatures::getTargetSignalDuration() const noexcept
{
    return pImpl->mChannelFeatures.getTargetSignalDuration();
}

int ThreeChannelFeatures::getTargetSignalLength() const noexcept
{
    return pImpl->mChannelFeatures.getTargetSignalLength();
}
