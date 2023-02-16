#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
#include "private/h5io.hpp"
#define EXPECTED_SIGNAL_LENGTH 400
#define SAMPLING_RATE 100
#define N_CHANNELS 1
#include "openvino.hpp"

using namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP;

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

/// Expected signal length
int Inference::getExpectedSignalLength() noexcept
{
    return EXPECTED_SIGNAL_LENGTH;
}

/// Sampling rate
double Inference::getSamplingRate() noexcept
{
    return SAMPLING_RATE;
}
