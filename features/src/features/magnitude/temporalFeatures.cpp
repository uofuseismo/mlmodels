#include "uuss/features/magnitude/temporalFeatures.hpp"

using namespace UUSS::Features::Magnitude;

class TemporalFeatures::TemporalFeaturesImpl
{
public:
    std::pair<double, double> mMinMax{0, 0};
    double mVariance{-1};
    bool mHaveMinMax{false};
};

/// C'tor
TemporalFeatures::TemporalFeatures() :
    pImpl(std::make_unique<TemporalFeaturesImpl> ())
{
}

/// Copy c'tor
TemporalFeatures::TemporalFeatures(const TemporalFeatures &features)
{
    *this = features;
}

/// Move c'tor
TemporalFeatures::TemporalFeatures(TemporalFeatures &&features) noexcept
{
    *this = std::move(features);
}

/// Copy assignment
TemporalFeatures& TemporalFeatures::operator=(const TemporalFeatures &features)
{
    if (&features == this){return *this;}
    pImpl = std::make_unique<TemporalFeaturesImpl> (*features.pImpl);
    return *this;
}

/// Move assignment
TemporalFeatures& TemporalFeatures::operator=(TemporalFeatures &&features) noexcept
{
    if (&features == this){return *this;}
    pImpl = std::move(features.pImpl);
    return *this;
}

/// Reset class
void TemporalFeatures::clear() noexcept
{
    pImpl = std::make_unique<TemporalFeaturesImpl> ();
}

/// Destructor
TemporalFeatures::~TemporalFeatures() = default;

/// Variance
void TemporalFeatures::setVariance(const double variance)
{
    if (variance < 0){throw std::invalid_argument("Variance is negative");}
    pImpl->mVariance = variance;
}

double TemporalFeatures::getVariance() const
{
    if (!haveVariance()){throw std::runtime_error("Variance not set");}
    return pImpl->mVariance;
}

bool TemporalFeatures::haveVariance() const noexcept
{
    return (pImpl->mVariance >= 0);
}

/// Min/max value
void TemporalFeatures::setMinimumAndMaximumValue(
    const std::pair<double, double> &minMax)
{
    if (minMax.first > minMax.second)
    {
        throw std::invalid_argument("minMax.first > minMax.second");
    }
    pImpl->mMinMax = minMax;
    pImpl->mHaveMinMax = true;
}

std::pair<double, double> TemporalFeatures::getMinimumAndMaximumValue() const
{
    if (!haveMinimumAndMaximumValue())
    {
        throw std::runtime_error("min/max not set");
    }
    return pImpl->mMinMax;
}

bool TemporalFeatures::haveMinimumAndMaximumValue() const noexcept
{
    return pImpl->mHaveMinMax;
}
