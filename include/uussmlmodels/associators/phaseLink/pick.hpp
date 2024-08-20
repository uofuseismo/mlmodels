#ifndef UUSS_MLMODELS_ASSOCIATORS_PHASE_LINK_PICK_HPP
#define UUSS_MLMODELS_ASSOCIATORS_PHASE_LINK_PICK_HPP
#include <chrono>
#include <string>
#include <optional>
#include <memory>
namespace UUSSMLModels::Associators::PhaseLink
{
/// @class Pick "pick.hpp" "uussmlmodels/associators/phaseLink/pick.hpp"
/// @brief Defines a pick to associate with PhaseLink.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Pick
{
public:
    /// @brief Defines the seismic phase to associate.
    enum class Phase
    {
        P, /*!< This is a P pick. */
        S  /*!< This is an S pick. */
    };
public:
    /// @name Constructors
    /// @{

    /// @brief Constructor.
    Pick();
    /// @brief Copy constructor.
    /// @param[in] pick  The pick from which to initialize this class.
    Pick(const Pick &pick);
    /// @brief Move constructor.
    /// @param[in,out] pick  The pick from which to initialize this class.
    ///                      On exit, pick's behavior is undefined.
    Pick(Pick &&pick) noexcept;
    /// @}

    /// @name Required Properties
    /// @{

    /// @brief Sets the network code.
    /// @param[in] network  The network code.
    void setNetwork(const std::string &network);
    /// @result The network code.
    /// @throws std::runtime_error if \c haveNetwork() is false.
    [[nodiscard]] std::string getNetwork() const;
    /// @result True indicates the network code was set.
    [[nodiscard]] bool haveNetwork() const noexcept;

    /// @brief Sets the station name.
    /// @param[in] station  The station name.
    void setStation(const std::string &station);
    /// @result The station name.
    /// @throws std::runtime_error if \c haveStation() is false.
    [[nodiscard]] std::string getStation() const;
    /// @result True indicates the station was set.
    [[nodiscard]] bool haveStation() const noexcept;

    /// @brief The arrival time (UTC) in seconds since the epoch.
    void setTime(double time) noexcept;
    /// @brief The arrival time (UTC) in microseconds since the epoch.
    void setTime(const std::chrono::microseconds &time) noexcept;
    /// @result The arrival time (UTC) in seconds since the epoch.
    /// @throws std::runtime_error if \c haveTime() is false.
    [[nodiscard]] double getTime() const;
    /// @result True indicates the time was set.
    [[nodiscard]] bool haveTime() const noexcept;

    /// @brief Sets the phase.
    /// @param[in] phase  The phase type.
    void setPhase(Phase phase) noexcept;
    /// @result The phase type.
    /// @throws std::runtime_error if \c havePhase() is false.
    [[nodiscard]] Phase getPhase() const;
    /// @result True indicates the phase was set.
    [[nodiscard]] bool havePhase() const noexcept;
    /// @}

    /// @name Optional
    /// @{
 
    /// @brief Sets the unique identifier for this pick.
    /// @param[in] identifier  The pick identifier.
    void setIdentifier(int64_t identifier) noexcept;
    /// @result The unique identifier.
    [[nodiscard]] std::optional<int64_t> getIdentifier() const noexcept;
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~Pick();
    /// @}

    /// @name Operators
    /// @{

    /// @brief Copy assignment.
    /// @param[in] pick  The pick to copy to this.
    /// @result A deep copy of the pick.
    Pick& operator=(const Pick &pick);
    /// @brief Move assignment.
    /// @param[in,out] pick  The memory from pick to move to this.
    ///                      On exit, pick's behavior is undefined.
    /// @result The memory from pick moved to this.
    Pick& operator=(Pick &&pick) noexcept;
    /// @}
private:
    class PickImpl;
    std::unique_ptr<PickImpl> pImpl;
};
}
#endif
