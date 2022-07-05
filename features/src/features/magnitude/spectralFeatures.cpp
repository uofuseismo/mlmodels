#include "uuss/features/magnitude/spectralFeatures.hpp"

using namespace UUSS::Features::Magnitude;

class SpectralFeatures::SpectralFeaturesImpl
{
public:
    std::pair<double, double> mDominantPeriodAndCumulativeAmplitude{0, 0};
    std::pair<double, double> mDominantPeriodAndAmplitude{0, 0};
    bool mHaveDominantPeriodAndCumulativeAmplitude{false};
    bool mHaveDominantPeriodAndAmplitude{false};
};

/// C'tor
SpectralFeatures::SpectralFeatures() :
    pImpl(std::make_unique<SpectralFeaturesImpl> ())
{
}

/// Copy c'tor
SpectralFeatures::SpectralFeatures(const SpectralFeatures &features)
{
    *this = features;
}

/// Move c'tor
SpectralFeatures::SpectralFeatures(SpectralFeatures &&features) noexcept
{
    *this = std::move(features);
}

/// Copy assignment
SpectralFeatures& SpectralFeatures::operator=(const SpectralFeatures &features)
{
    if (&features == this){return *this;}
    pImpl = std::make_unique<SpectralFeaturesImpl> (*features.pImpl);
    return *this;
}

/// Move assignment
SpectralFeatures&
SpectralFeatures::operator=(SpectralFeatures &&features) noexcept
{
    if (&features == this){return *this;}
    pImpl = std::move(features.pImpl);
    return *this;
}

/// The dominant period/amplitude
void SpectralFeatures::setDominantPeriodAndAmplitude(
    const std::pair<double, double> &periodAmplitude)
{
    if (periodAmplitude.first <= 0)
    {
        throw std::invalid_argument("Period must be positive");
    }
    if (periodAmplitude.second <= 0)
    {
        throw std::invalid_argument("Amplitude cannot be negative");
    }
    pImpl->mDominantPeriodAndAmplitude = periodAmplitude;
    pImpl->mHaveDominantPeriodAndAmplitude = true;
}

bool SpectralFeatures::haveDominantPeriodAndAmplitude() const noexcept
{
    return pImpl->mHaveDominantPeriodAndAmplitude;
}

/// The dominant period/cumulative amplitude
void SpectralFeatures::setDominantPeriodAndCumulativeAmplitude(
    const std::pair<double, double> &periodAmplitude)
{
    if (periodAmplitude.first <= 0)
    {   
        throw std::invalid_argument("Period must be positive");
    }   
    if (periodAmplitude.second <= 0)
    {   
        throw std::invalid_argument("Cumulative amplitude cannot be negative");
    }   
    pImpl->mDominantPeriodAndCumulativeAmplitude = periodAmplitude;
    pImpl->mHaveDominantPeriodAndCumulativeAmplitude = true;
}

bool SpectralFeatures::haveDominantPeriodAndCumulativeAmplitude() const noexcept
{
    return pImpl->mHaveDominantPeriodAndCumulativeAmplitude;
}

/// Clear
void SpectralFeatures::clear() noexcept
{
    pImpl = std::make_unique<SpectralFeaturesImpl> ();
}

/// Destructor
SpectralFeatures::~SpectralFeatures() = default;
