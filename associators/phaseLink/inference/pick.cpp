#include <string>
#include <cmath>
#include <algorithm>
#include "uussmlmodels/associators/phaseLink/pick.hpp"

namespace
{

/// @result True indicates that the string is empty or full of blanks.
[[maybe_unused]] [[nodiscard]]
bool isEmpty(const std::string &s) 
{
    if (s.empty()){return true;}
    return std::all_of(s.begin(), s.end(), [](const char c)
                       {
                           return std::isspace(c);
                       });
}

[[nodiscard]]
std::string convertString(const std::string &input)
{
    auto result = input;
    std::remove_if(result.begin(), result.end(), ::isspace);
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

}

using namespace UUSSMLModels::Associators::PhaseLink;

class Pick::PickImpl
{
public:
    std::string mNetwork;
    std::string mStation;
    double mTime;
    Pick::Phase mPhase;
    int64_t mIdentifier{0};
    bool mHavePhase{false};
    bool mHaveTime{false};
    bool mHaveIdentifier{false};
};

/// Constructor
Pick::Pick() :
    pImpl(std::make_unique<PickImpl> ())
{
}

/// Copy constructor
Pick::Pick(const Pick &pick)
{
    *this = pick;
}

/// Move constructor
Pick::Pick(Pick &&pick) noexcept
{
    *this = std::move(pick);
}

/// Copy assignment
Pick& Pick::operator=(const Pick &pick)
{
    if (&pick == this){return *this;}
    pImpl = std::make_unique<PickImpl> (*pick.pImpl);
    return *this;
}

/// Move assignment
Pick& Pick::operator=(Pick &&pick) noexcept
{
    if (&pick == this){return *this;}
    pImpl = std::move(pick.pImpl);
    return *this;
}

/// Destructor
Pick::~Pick() = default;

/// Reset class.
void Pick::clear() noexcept
{
    pImpl = std::make_unique<PickImpl> ();
}

/// Network
void Pick::setNetwork(const std::string &s)
{
    auto network = ::convertString(s); 
    if (::isEmpty(network)){throw std::invalid_argument("Network is empty");}
    pImpl->mNetwork = network;
}

std::string Pick::getNetwork() const
{
    if (!haveNetwork()){throw std::runtime_error("Network not set yet");}
    return pImpl->mNetwork;
}

bool Pick::haveNetwork() const noexcept
{
    return !pImpl->mNetwork.empty();
}

/// Station
void Pick::setStation(const std::string &s)
{
    auto station = ::convertString(s); 
    if (::isEmpty(station)){throw std::invalid_argument("Station is empty");}
    pImpl->mStation = station;
}

std::string Pick::getStation() const
{
    if (!haveStation()){throw std::runtime_error("Station not set yet");}
    return pImpl->mStation;
}

bool Pick::haveStation() const noexcept
{
    return !pImpl->mStation.empty();
}

/// Pick time
void Pick::setTime(const double time) noexcept
{
    pImpl->mTime = time;
    pImpl->mHaveTime = true;
}

void Pick::setTime(const std::chrono::microseconds &time) noexcept
{
    auto pickTime = time.count()*1.e-6;
    setTime(pickTime);
}

double Pick::getTime() const
{
    if (!haveTime()){throw std::runtime_error("Time not yet set");}
    return pImpl->mTime;
}

bool Pick::haveTime() const noexcept
{
    return pImpl->mHaveTime;
}

/// Phase
void Pick::setPhase(const Phase phase) noexcept
{
    pImpl->mPhase = phase;
    pImpl->mHavePhase = true;
}

Pick::Phase Pick::getPhase() const
{
    if (!havePhase()){throw std::runtime_error("Phase not set");}
    return pImpl->mPhase;
}

bool Pick::havePhase() const noexcept
{
    return pImpl->mHavePhase;
}

/// Identifier
void Pick::setIdentifier(const int64_t identifier) noexcept
{
    pImpl->mIdentifier = identifier;
    pImpl->mHaveIdentifier = true;
}

std::optional<int64_t> Pick::getIdentifier() const noexcept
{
    return pImpl->mHaveIdentifier ? std::optional<int64_t> (pImpl->mIdentifier) :
                                    std::nullopt;
}
