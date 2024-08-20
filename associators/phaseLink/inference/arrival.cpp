#include <string>
#include <cmath>
#include <algorithm>
#include "uussmlmodels/associators/phaseLink/arrival.hpp"
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

class Arrival::ArrivalImpl
{
public:
    std::string mNetwork;
    std::string mStation;
    double mTime;
    double mProbability;
    Arrival::Phase mPhase;
    int64_t mIdentifier{0};
    bool mHavePhase{false};
    bool mHaveTime{false};
    bool mHaveIdentifier{false};
    bool mHaveProbability{false};
};

/// Constructor
Arrival::Arrival() :
    pImpl(std::make_unique<ArrivalImpl> ())
{
}

/// Construtor
Arrival::Arrival(const Pick &pick) :
    pImpl(std::make_unique<ArrivalImpl> ())
{
    Arrival arrival; 
    if (pick.haveNetwork()){arrival.setNetwork(pick.getNetwork());}
    if (pick.haveStation()){arrival.setStation(pick.getStation());}
    if (pick.haveTime()){arrival.setTime(pick.getTime());}
    if (pick.havePhase())
    {
        if (pick.getPhase() == Pick::Phase::P)
        {
            arrival.setPhase(Arrival::Phase::P);
        }
        else if (pick.getPhase() == Pick::Phase::S)
        {
            arrival.setPhase(Arrival::Phase::S);
        }
    }
    auto identifier = pick.getIdentifier();
    if (identifier){arrival.setIdentifier(*identifier);}
    *this = std::move(arrival);
}

/// Copy constructor
Arrival::Arrival(const Arrival &arrival)
{
    *this = arrival;
}

/// Move constructor
Arrival::Arrival(Arrival &&arrival) noexcept
{
    *this = std::move(arrival);
}

/// Copy assignment
Arrival& Arrival::operator=(const Arrival &arrival)
{
    if (&arrival == this){return *this;}
    pImpl = std::make_unique<ArrivalImpl> (*arrival.pImpl);
    return *this;
}

/// Move assignment
Arrival& Arrival::operator=(Arrival &&arrival) noexcept
{
    if (&arrival == this){return *this;}
    pImpl = std::move(arrival.pImpl);
    return *this;
}

/// Destructor
Arrival::~Arrival() = default;

/// Reset class.
void Arrival::clear() noexcept
{
    pImpl = std::make_unique<ArrivalImpl> ();
}

/// Network
void Arrival::setNetwork(const std::string &s)
{
    auto network = ::convertString(s); 
    if (::isEmpty(network)){throw std::invalid_argument("Network is empty");}
    pImpl->mNetwork = network;
}

std::string Arrival::getNetwork() const
{
    if (!haveNetwork()){throw std::runtime_error("Network not set yet");}
    return pImpl->mNetwork;
}

bool Arrival::haveNetwork() const noexcept
{
    return !pImpl->mNetwork.empty();
}

/// Station
void Arrival::setStation(const std::string &s)
{
    auto station = ::convertString(s); 
    if (::isEmpty(station)){throw std::invalid_argument("Station is empty");}
    pImpl->mStation = station;
}

std::string Arrival::getStation() const
{
    if (!haveStation()){throw std::runtime_error("Station not set yet");}
    return pImpl->mStation;
}

bool Arrival::haveStation() const noexcept
{
    return !pImpl->mStation.empty();
}

/// Arrival time
void Arrival::setTime(const double time) noexcept
{
    pImpl->mTime = time;
    pImpl->mHaveTime = true;
}

void Arrival::setTime(const std::chrono::microseconds &time) noexcept
{
    auto arrivalTime = time.count()*1.e-6;
    setTime(arrivalTime);
}

double Arrival::getTime() const
{
    if (!haveTime()){throw std::runtime_error("Time not yet set");}
    return pImpl->mTime;
}

bool Arrival::haveTime() const noexcept
{
    return pImpl->mHaveTime;
}

/// Phase
void Arrival::setPhase(const Phase phase) noexcept
{
    pImpl->mPhase = phase;
    pImpl->mHavePhase = true;
}

Arrival::Phase Arrival::getPhase() const
{
    if (!havePhase()){throw std::runtime_error("Phase not set");}
    return pImpl->mPhase;
}

bool Arrival::havePhase() const noexcept
{
    return pImpl->mHavePhase;
}

/// Identifier
void Arrival::setIdentifier(const int64_t identifier) noexcept
{
    pImpl->mIdentifier = identifier;
    pImpl->mHaveIdentifier = true;
}

std::optional<int64_t> Arrival::getIdentifier() const noexcept
{
    return pImpl->mHaveIdentifier ? std::optional<int64_t>(pImpl->mIdentifier) :
                                    std::nullopt;
}

/// Posterior probablity
void Arrival::setProbability(double probability) 
{
    if (probability < 0 || probability > 1)
    {
        throw std::invalid_argument("Probability must be in range [0,1]");
    }
    pImpl->mProbability = probability;
    pImpl->mHaveProbability = true;
}

std::optional<double> Arrival::getProbability() const noexcept
{
    return pImpl->mHaveProbability ? std::optional<double> (pImpl->mProbability) :
                                     std::nullopt;
}
