#include <string>
#include <algorithm>
#include "uuss/features/magnitude/channel.hpp"
#include "private/shiftLongitude.hpp"

using namespace UUSS::Features::Magnitude;

class Channel::ChannelImpl
{
public:
    std::string mUnits;
    std::string mNetworkCode;
    std::string mStationCode;
    std::string mChannelCode;
    std::string mLocationCode{"01"};
    double mSamplingRate{0};
    double mSimpleResponseValue{0};
    double mLatitude{-1000};
    double mLongitude{0};
    double mAzimuth{-1000};
    bool mHaveLongitude{false};
};

/// C'tor
Channel::Channel() :
    pImpl(std::make_unique<ChannelImpl> ()) 
{
}

/// Copy c'tor
Channel::Channel(const Channel &channel)
{
    *this = channel;
}

/// Move c'tor
Channel::Channel(Channel &&channel) noexcept
{
    *this = std::move(channel);
}

/// Copy assignment
Channel& Channel::operator=(const Channel &channel)
{
    if (&channel == this){return *this;}
    pImpl = std::make_unique<ChannelImpl> (*channel.pImpl);
    return *this;
}

/// Move assignment
Channel& Channel::operator=(Channel &&channel) noexcept
{
    if (&channel == this){return *this;}
    pImpl = std::move(channel.pImpl);
    return *this;
}

/// Reset class
void Channel::clear() noexcept
{
    pImpl = std::make_unique<ChannelImpl> (); 
}

/// Destructor
Channel::~Channel() = default;

/// Latitude
void Channel::setLatitude(const double latitude)
{
    if (latitude < -90 || latitude > 90) 
    {
        throw std::invalid_argument("Latitude must be in range [-90,90]");
    }
    pImpl->mLatitude = latitude;
}

double Channel::getLatitude() const
{
    if (!haveLatitude()){throw std::runtime_error("Channel latitude not set");}
    return pImpl->mLatitude;
}

bool Channel::haveLatitude() const noexcept
{
    return (pImpl->mLatitude > -1000);
}

/// Longitude
void Channel::setLongitude(const double longitude) noexcept
{
    pImpl->mLongitude = shiftLongitude(longitude);
    pImpl->mHaveLongitude = true;
}

double Channel::getLongitude() const
{
    if (!haveLongitude())
    {
        throw std::runtime_error("Channel longitude not set");
    }
    return pImpl->mLongitude;
}

bool Channel::haveLongitude() const noexcept
{
    return pImpl->mHaveLongitude;
}

/// Sampling rate
void Channel::setSamplingRate(const double samplingRate)
{
    if (samplingRate <= 0)
    {   
        throw std::invalid_argument("Sampling rate must be positive");
    }   
    pImpl->mSamplingRate = samplingRate;
}

double Channel::getSamplingRate() const
{
    if (!haveSamplingRate()){throw std::runtime_error("Sampling rate not set");}
    return pImpl->mSamplingRate;
}

bool Channel::haveSamplingRate() const noexcept
{
    return (pImpl->mSamplingRate > 0);
}

/// Simple response
void Channel::setSimpleResponse(const double simpleResponse,
                                const std::string &unitsIn)
{
    if (simpleResponse == 0)
    {   
        throw std::invalid_argument("Simple response cannot be zero");
    }   
    std::string units{unitsIn};
    std::transform(unitsIn.begin(), unitsIn.end(), units.begin(), ::toupper);
    if (units != "DU/M/S**2" && units != "DU/M/S")
    {
        throw std::runtime_error("units = " + unitsIn + " not handled");
    }
    pImpl->mUnits = unitsIn;
    pImpl->mSimpleResponseValue = simpleResponse;
}

double Channel::getSimpleResponseValue() const
{
    if (!haveSimpleResponse())
    {
        throw std::runtime_error("Simple response not set");
    }
    return pImpl->mSimpleResponseValue;
}

std::string Channel::getSimpleResponseUnits() const
{
    if (!haveSimpleResponse())
    {
        throw std::runtime_error("Simple response not set");
    }
    return pImpl->mUnits;
}

bool Channel::haveSimpleResponse() const noexcept
{
    return !pImpl->mUnits.empty();
}

/// Component azimuth
void Channel::setAzimuth(const double azimuth)
{
    if (azimuth < 0 || azimuth >= 360)
    {
        throw std::invalid_argument("Azimuth must be in range [0,360)");
    }
    pImpl->mAzimuth = azimuth;
}

double Channel::getAzimuth() const
{
    if (!haveAzimuth()){throw std::runtime_error("Channel azimuth not set");}
    return pImpl->mAzimuth;
}

bool Channel::haveAzimuth() const noexcept
{
    return (pImpl->mAzimuth > -1000);
}

/// Network
void Channel::setNetworkCode(const std::string &network)
{
    pImpl->mNetworkCode = network;
}

std::string Channel::getNetworkCode() const
{
    return pImpl->mNetworkCode;
}

/// Station code
void Channel::setStationCode(const std::string &station)
{
    pImpl->mStationCode = station;
}

std::string Channel::getStationCode() const
{
    return pImpl->mStationCode;
}

/// Channel code
void Channel::setChannelCode(const std::string &channel)
{
    pImpl->mChannelCode = channel;
}

std::string Channel::getChannelCode() const
{
    return pImpl->mChannelCode;
}

/// Location code
void Channel::setLocationCode(const std::string &locationCode)
{
    pImpl->mLocationCode = locationCode;
}

std::string Channel::getLocationCode() const noexcept
{
    return pImpl->mLocationCode;
}


