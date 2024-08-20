#ifndef UUSS_MLMODELS_ASSOCIATORS_PHASE_LINK_ARRIVAL_HPP
#define UUSS_MLMODELS_ASSOCIATORS_PHASE_LINK_ARRIVAL_HPP
#include <chrono>
#include <string>
#include <optional>
#include <memory>
namespace UUSSMLModels::Associators::PhaseLink
{
 class Pick;
}
namespace UUSSMLModels::Associators::PhaseLink
{
/// @class Arrival "arrival.hpp" "uussmlmodels/associators/phaseLink/arrival.hpp"
/// @brief Defines an arrival that has been associated with PhaseLink.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Arrival
{
public:
    /// @brief Defines the phase of the associated arrival.
    enum class Phase
    {
        P, /*!< This is a P arrival. */
        S  /*!< This is an S arrival. */
    };
public:
    /// @name Constructors
    /// @{

    /// @brief Constructor.
    Arrival();
    /// @brief Creates the arrival from a pick.
    /// @param[in] pick  The pick from which to create this arrival.
    explicit Arrival(const Pick &pick);
    /// @brief Copy constructor.
    /// @param[in] arrival  The arrival from which to initialize this class.
    Arrival(const Arrival &arrival);
    /// @brief Move constructor.
    /// @param[in,out] arrival  The arrival from which to initialize this class.
    ///                         On exit, arrival's behavior is undefined.
    Arrival(Arrival &&arrival) noexcept;
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
 
    /// @brief Sets the unique identifier for this arrival.
    /// @param[in] identifier  The arrival identifier.
    void setIdentifier(int64_t identifier) noexcept;
    /// @result The unique identifier.
    [[nodiscard]] std::optional<int64_t> getIdentifier() const noexcept;

    /// @brief The posterior probability of this classification.
    /// @param[in] probability   The posterior probability of this being an
    ///                          arrival associated with the root arrival. 
    /// @throws std::invalid_argument if probability not in range [0,1]. 
    void setProbability(double probability);
    /// @result The posterior probability.
    [[nodiscard]] std::optional<double> getProbability() const noexcept;
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~Arrival();
    /// @}

    /// @name Operators
    /// @{

    /// @brief Copy assignment.
    /// @param[in] arrival  The arrival to copy to this.
    /// @result A deep copy of the arrival.
    Arrival& operator=(const Arrival &arrival);
    /// @brief Move assignment.
    /// @param[in,out] arrival  The memory from arrival to move to this.
    ///                         On exit, arrival's behavior is undefined.
    /// @result The memory from arrival moved to this.
    Arrival& operator=(Arrival &&arrival) noexcept;
    /// @}
private:
    class ArrivalImpl;
    std::unique_ptr<ArrivalImpl> pImpl;
};
}
#endif
