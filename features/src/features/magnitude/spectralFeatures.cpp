#include "uuss/features/magnitude/spectralFeatures.hpp"

using namespace UUSS::Features::Magnitude;

class SpectralFeatures::SpectralFeaturesImpl
{
public:
    std::vector<std::pair<double, double>> mAverageFrequencyAmplitude;
    std::pair<double, double> mDominantFrequencyAndAmplitude{0, 0};
    bool mHaveDominantFrequencyAndAmplitude{false};
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
void SpectralFeatures::setDominantFrequencyAndAmplitude(
    const std::pair<double, double> &frequencyAmplitude)
{
    if (frequencyAmplitude.first <= 0)
    {
        throw std::invalid_argument("Frequency must be positive");
    }
    if (frequencyAmplitude.second <= 0)
    {
        throw std::invalid_argument("Amplitude cannot be negative");
    }
    pImpl->mDominantFrequencyAndAmplitude = frequencyAmplitude;
    pImpl->mHaveDominantFrequencyAndAmplitude = true;
}

bool SpectralFeatures::haveDominantFrequencyAndAmplitude() const noexcept
{
    return pImpl->mHaveDominantFrequencyAndAmplitude;
}

void SpectralFeatures::setAverageFrequenciesAndAmplitudes(
    const std::vector<std::pair<double, double>> &frequencyAmplitude)
{
    if (frequencyAmplitude.empty())
    {
        throw std::invalid_argument("Frequency amplitude empty");
    }
    for (const auto &fa : frequencyAmplitude)
    {
        if (fa.first <= 0)
        {
            throw std::invalid_argument("Frequency must be positive");
        }
        if (fa.second < 0)
        {
           throw std::invalid_argument("Amplitude must be non-negative");
        }
    }
    pImpl->mAverageFrequencyAmplitude = frequencyAmplitude;
}

std::vector<std::pair<double, double>>
SpectralFeatures::getAverageFrequenciesAndAmplitudes() const
{
    if (!haveAverageFrequenciesAndAmplitudes())
    {
        throw std::runtime_error("Average frequency/amplitude not set");
    }
    return pImpl->mAverageFrequencyAmplitude;
}

bool SpectralFeatures::haveAverageFrequenciesAndAmplitudes() const noexcept
{
    return !pImpl->mAverageFrequencyAmplitude.empty();
}

/*
/// The dominant frequency/cumulative amplitude
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
*/

/// Clear
void SpectralFeatures::clear() noexcept
{
    pImpl = std::make_unique<SpectralFeaturesImpl> ();
}

/// Destructor
SpectralFeatures::~SpectralFeatures() = default;
