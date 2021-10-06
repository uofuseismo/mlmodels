#include <string>
#include <vector>
#include "localMagnitudeProcessing.hpp"
#include "uuss/amplitudes/localMagnitudeProcessing.hpp"
#include <pybind11/stl.h>

using namespace PUUSSMLModels::Amplitudes;

/// C'tor
LocalMagnitudeProcessing::LocalMagnitudeProcessing() :
    pImpl(std::make_unique<UUSS::Amplitudes::LocalMagnitudeProcessing> ())
{
}

/// Destructor
LocalMagnitudeProcessing::~LocalMagnitudeProcessing() = default;

std::vector<double> LocalMagnitudeProcessing::processWaveform(
    const bool isVelocity, const double gain,
    const std::vector<double> &x,
    const double samplingPeriod)
{
/*
    if (channel.size() != 3)
    {
        throw std::invalid_argument("Channel.size should be length 3");
    }
*/
    if (gain == 0){throw std::invalid_argument("Gain is zero");}
    if (samplingPeriod <= 0)
    {
        throw std::invalid_argument("Sampling rate = "
                                  + std::to_string(samplingPeriod)
                                  + " must be positive");
    }
    std::vector<double> y;
/*
    bool isVelocity = true;
    if (channel[1] == 'N' || channel[1] == 'n')
    {
        isVelocity = false;
    }
    else if (channel[1] == 'H' || channel[1] == 'h')
    {
        isVelocity = true;
    }
    else
    {
        throw std::invalid_argument("Unhandled channel: " + channel);
    }
*/
    pImpl->processWaveform(isVelocity, gain, x.size(),
                           samplingPeriod, x.data(), &y);
    return y;
}

/// Get the start sampling period 
double LocalMagnitudeProcessing::getTargetSamplingPeriod() const noexcept
{
    return pImpl->getTargetSamplingPeriod();
}

std::vector<int> LocalMagnitudeProcessing::computeMinMaxSignal(
    const std::vector<double> &x)
{
    std::vector<int> result;
    pImpl->computeMinMaxSignal(x, &result);
    return result;
}

/// Create class
void PUUSSMLModels::Amplitudes::initializeLocalMagnitudeProcessing(
    pybind11::module &m)
{
    pybind11::class_<LocalMagnitudeProcessing> p(m, "LocalMagnitudeProcessing");
    p.def(pybind11::init<> ());
    p.doc() = "Performs the Wood-Anderson filtering in Jiggle which is applied prior to the analysts picking waveform amplitudes for local magnitude calculations.";

    p.def("process_waveform",
          &LocalMagnitudeProcessing::processWaveform,
          "Performs the appropriate preprocessing to the waveform/channel/gain with the given sampling period in seconds.  The gain should result in the resulting amplitude of meters/second or meters/second/second.");
    p.def("compute_min_max_signal",
          &LocalMagnitudeProcessing::computeMinMaxSignal,
          "From the processed waveform this computes a signal that is 1 for a local maximum, -1 for a local minimum, and 0 otherwise");
    p.def_property_readonly("target_sampling_period",
                            &LocalMagnitudeProcessing::getTargetSamplingPeriod,
                            "The sampling period of the processed waveform in seconds.");
}

