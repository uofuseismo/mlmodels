#include <string>
#include <vector>
#include "inference.hpp"

#define SAMPLING_RATE 100
#define MINIMUM_SIGNAL_LENGTH 1008

using namespace UUSSMLModels::Detectors::UNetThreeComponentP;

class Inference::InferenceImpl
{
public:

};

/// Constructor
Inference::Inference() :
    pImpl(std::make_unique<InferenceImpl> ())
{
}

/// Destructor
Inference::~Inference() = default;

/// Sampling rate
double Inference::getSamplingRate() noexcept
{ 
    return SAMPLING_RATE;
}

/// Minimum signal length
int Inference::getMinimumSignalLength() noexcept
{
    return MINIMUM_SIGNAL_LENGTH;
}
