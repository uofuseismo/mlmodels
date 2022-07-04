#ifndef UUSS_FEATURES_MAGNITUDE_HYPOCENTER_HPP
#define UUSS_FEATURES_MAGNITUDE_HYPOCENTER_HPP
#include <memory>
namespace UUSS::Features::Magnitude
{
/// @class Hypocenter "hypocenter.hpp" "uuss/features/magnitude/hypocenter.hpp"
/// @brief Defines the event hypocenter.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Hypocenter
{
public:
    /// @name Constructors
    /// @{

    /// @brief Constructor.
    Hypocenter();
    /// @brief Copy constructor.
    /// @param[in] hypocenter  The hypocenter from which to initialize this
    ///                        class.
    Hypocenter(const Hypocenter &hypocenter);
    /// @brief Move constructor.
    /// @param[in,out] hypocenter  The hypocenter from which to initialize
    ///                            this class.  On exit, hypocenter's behavior
    ///                            is undefined.
    Hypocenter(Hypocenter &&hypocenter) noexcept;
    /// @}

    /// @name Operators
    /// @{
 
    /// @brief Copy assignment.
    /// @param[in] hypocenter  The hypocenter to copy to this. 
    /// @result The memory from hypocenter moved to this.
    Hypocenter& operator=(const Hypocenter &hypocenter);
    /// @brief Move assignment.
    /// @param[in,out] hypocenter  The hypocenter whose memory will be moved
    ///                            to this.  On exit, hypocenter's behavior
    ///                            is undefined.
    /// @result The memory from hypocenter moved to this.
    Hypocenter& operator=(Hypocenter &&hypocenter) noexcept;
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

    /// @brief Sets the event's depth.
    /// @param[in] depth  The event depth in kilometers.
    void setDepth(double depth) noexcept;
    /// @result The event depth in kilometers.
    /// @throws std::runtime_error if the event depth was not set.
    [[nodiscard]] double getDepth() const;
    /// @result True indicates the event depth was set.
    [[nodiscard]] bool haveDepth() const noexcept;
    /// @}

    /// @brief Optional Information
    /// @{

    /// @brief Sets the event identifier.
    void setEventIdentifier(int64_t eventIdentifier) noexcept;
    /// @result The event identifier.
    [[nodiscard]] int64_t getEventIdentifier() const noexcept;     
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~Hypocenter();
    /// @}
private:
    class HypocenterImpl;
    std::unique_ptr<HypocenterImpl> pImpl;
};
}
#endif
