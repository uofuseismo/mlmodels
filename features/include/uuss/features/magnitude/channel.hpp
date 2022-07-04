#ifndef UUSS_FEATURES_MAGNITUDE_CHANNEL_HPP
#define UUSS_FEATURES_MAGNITUDE_CHANNEL_HPP
#include <memory>
namespace UUSS::Features::Magnitude
{
/// @class Channel "channel.hpp" "uuss/features/magnitude/channel.hpp"
/// @brief Defines some necessary information for a station's channel data to
///        be used in a magnitude feature extraction calculation.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Channel
{
public:
    /// @name Constructors
    /// @{

    /// @brief Constructor.
    Channel();
    /// @brief Copy constructor.
    /// @param[in] channel  The channel from which to initialize this
    ///                     class.
    Channel(const Channel &channel);
    /// @brief Move constructor.
    /// @param[in,out] channel  The channel from which to initialize
    ///                         this class.  On exit, channel's behavior
    ///                         is undefined.
    Channel(Channel &&channel) noexcept;
    /// @}

    /// @name Operators
    /// @{
 
    /// @brief Copy assignment.
    /// @param[in] channel  The channel to copy to this. 
    /// @result The memory from channel moved to this.
    Channel& operator=(const Channel &channel);
    /// @brief Move assignment.
    /// @param[in,out] channel  The channel whose memory will be moved
    ///                         to this.  On exit, channel's behavior
    ///                         is undefined.
    /// @result The memory from channel moved to this.
    Channel& operator=(Channel &&channel) noexcept;
    /// @}

    /// @brief Required Parameters
    /// @{

    /// @brief Sets the event's latitude.
    /// @param[in] latitude  The event latitude in degrees.
    /// @throws std::invalid_argument if the latitude is not in range [-90,90].
    void setLatitude(double latitude);
    /// @result The event latitude in degrees.
    /// @throws std::runtime_error if the depth was not set.
    [[nodiscard]] double getLatitude() const;
    /// @result True indicates the event latitude was set.
    [[nodiscard]] bool haveLatitude() const noexcept;

    /// @brief Sets the event's longitude.
    /// @param[in] longitude  The event longitude in degrees measured
    ///                       positive east.
    void setLongitude(double longitude) noexcept;
    /// @result The longitude in degrees.  Note, this will be in the range
    ///         [-180,180).
    [[nodiscard]] double getLongitude() const;
    /// @result True indicates the longitude was set.
    [[nodiscard]] bool haveLongitude() const noexcept;

    /// @brief Sets the channel's nominal sampling rate.
    /// @param[in] samplingRate  The sampling rate in Hz.
    /// @throws std::invalid_argument if the sampling rate is not positive.
    void setSamplingRate(double samplingRate);
    /// @result The sampling rate in Hz.
    /// @throws std::runtime_error if \c haveSamplingRate() is false.
    [[nodiscard]] double getSamplingRate() const;
    /// @result True indicates the sampling rate was set.
    [[nodiscard]] bool haveSamplingRate() const noexcept;

    /// @brief Sets the simple response units and its corresponding units.
    /// @param[in] value  The simple response value.  When the input signal
    ///                   is divided by this value then the result
    ///                   will be in M/S or M/S**2. 
    /// @param[in] units  The units.  This is from the AQMS database. 
    ///                   This can be DU/M/S for velocity or DU/M/S**2
    ///                   for acceleration.
    /// @throws std::runtime_error if the simple response is zero or the given
    ///         units are unhandled.
    void setSimpleResponse(double value, const std::string &units);
    /// @result The simple response value.
    /// @throw std::runtime_error if \c haveSimpleResponse() is false.
    [[nodiscard]] double getSimpleResponseValue() const;
    /// @result The simple response units.
    /// @throw std::runtime_error if \c haveSimpleResponse() is false.
    [[nodiscard]] std::string getSimpleResponseUnits() const;
    /// @result True indicates the simple response was set.
    [[nodiscard]] bool haveSimpleResponse() const noexcept;
    /// @}

    /// @brief Optional Information
    /// @{

    /// @brief Sets the channel's azimuth.  For example, this will be 0 for
    ///        north and vertical channels while this will be 90 for east
    ///        channels.
    /// @param[in] azimuth  The channel's azimuth in degrees measured positive
    ///                     east of north. 
    /// @throws std::invalid_arugment if the azimuth is not in the
    ///         range [0,360).
    void setAzimuth(double azimuth);
    /// @result The channel's azimuth in degrees.
    [[nodiscard]] double getAzimuth() const;
    /// @result True indicates the azimuth was set.
    [[nodiscard]] bool haveAzimuth() const noexcept;

    /// @brief Sets the network code.
    /// @param[in] network  The network code - e.g., UU.
    void setNetworkCode(const std::string &network);
    /// @result The network code.
    [[nodiscard]] std::string getNetworkCode() const;

    /// @brief Sets the station code. 
    /// @param[in] station  The station code - e.g., SRU. 
    void setStationCode(const std::string &station);
    /// @result The station code.
    [[nodiscard]] std::string getStationCode() const;

    /// @brief Sets the channel code.
    /// @param[in] channelCode  The channel code - e.g., ENZ for high-sample
    ///                         rate strong motion instruments that are
    ///                         proportional to acceleration or HHZ for
    ///                         high-sample rate broadband instruments that
    ///                         are proportional to velocity.
    void setChannelCode(const std::string &channelCode);
    /// @result The channel code.
    [[nodiscard]] std::string getChannelCode() const;

    /// @brief Sets the location code.
    /// @param[in] locationCode  The location code - e.g., 01.
    void setLocationCode(const std::string &locationCode);
    /// @result The location code.
    [[nodiscard]] std::string getLocationCode() const noexcept;
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~Channel();
    /// @}
private:
    class ChannelImpl;
    std::unique_ptr<ChannelImpl> pImpl;
};
}
#endif
