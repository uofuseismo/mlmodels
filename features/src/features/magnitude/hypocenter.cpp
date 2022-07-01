#include "uuss/features/magnitude/hypocenter.hpp"
#include "private/shiftLongitude.hpp"

using namespace UUSS::Features::Magnitude;

class Hypocenter::HypocenterImpl
{
public:
    double mLatitude{-1000};
    double mLongitude{0};
    double mDepth{0};
    int64_t mEventIdentifier{0};
    bool mHaveLongitude{false};
    bool mHaveDepth{false};
};

/// C'tor
Hypocenter::Hypocenter() :
    pImpl(std::make_unique<HypocenterImpl> ())
{
}

/// Copy c'tor
Hypocenter::Hypocenter(const Hypocenter &hypocenter)
{
    *this = hypocenter;
}

/// Move c'tor
Hypocenter::Hypocenter(Hypocenter &&hypocenter) noexcept
{
    *this = std::move(hypocenter);
}

/// Copy assignment
Hypocenter& Hypocenter::operator=(const Hypocenter &hypocenter)
{
    if (&hypocenter == this){return *this;}
    pImpl = std::make_unique<HypocenterImpl> (*hypocenter.pImpl);
    return *this;
}

/// Move assignment
Hypocenter& Hypocenter::operator=(Hypocenter &&hypocenter) noexcept
{
    if (&hypocenter == this){return *this;}
    pImpl = std::move(hypocenter.pImpl);
    return *this;
}

/// Reset class
void Hypocenter::clear() noexcept
{
    pImpl = std::make_unique<HypocenterImpl> ();
}

/// Destructor
Hypocenter::~Hypocenter() = default;

/// Latitude
void Hypocenter::setLatitude(const double latitude)
{
    if (latitude < -90 || latitude > 90)
    {
        throw std::invalid_argument("Latitude must be in range [-90,90]");
    }
    pImpl->mLatitude = latitude;
}

double Hypocenter::getLatitude() const
{
    if (!haveLatitude()){throw std::runtime_error("Latitude not set");}
    return pImpl->mLatitude;
}

bool Hypocenter::haveLatitude() const noexcept
{
    return (pImpl->mLatitude > -1000);
}

/// Longitude
void Hypocenter::setLongitude(const double longitude) noexcept
{
    pImpl->mLongitude = shiftLongitude(longitude);
    pImpl->mHaveLongitude = true;
}

double Hypocenter::getLongitude() const
{
    if (!haveLongitude()){throw std::runtime_error("Longitude not set");}
    return pImpl->mLongitude;
}

bool Hypocenter::haveLongitude() const noexcept
{
    return pImpl->mHaveLongitude;
}

/// Depth
void Hypocenter::setDepth(const double depth) noexcept
{
    pImpl->mDepth = depth;
    pImpl->mHaveDepth = true;
}

double Hypocenter::getDepth() const
{
    if (!haveDepth()){throw std::runtime_error("Depth not set");}
    return pImpl->mDepth;
}

bool Hypocenter::haveDepth() const noexcept
{
    return pImpl->mHaveDepth;
}

/// Event identifier
void Hypocenter::setEventIdentifier(const int64_t eventIdentifier) noexcept
{
    pImpl->mEventIdentifier = eventIdentifier;
}

int64_t Hypocenter::getEventIdentifier() const noexcept
{
    return pImpl->mEventIdentifier;
}
