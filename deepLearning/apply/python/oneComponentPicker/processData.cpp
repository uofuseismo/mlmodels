#include <uuss/oneComponentPicker/zcnn/processData.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "processData.hpp"

using namespace PUUSSMLModels::OneComponentPicker::ZCNN;

/// C'tor
ProcessData::ProcessData() :
    pImpl(std::make_unique<UUSS::OneComponentPicker::ZCNN::ProcessData> ())
{
}

/// Destructor
ProcessData::~ProcessData() = default;

std::vector<double> ProcessData::processWaveform(const std::vector<double> &x, 
                                                 const double samplingPeriod)
{
    if (samplingPeriod <= 0)
    {
        throw std::invalid_argument("Sampling rate = "
                                  + std::to_string(samplingPeriod)
                                  + " must be positive");
    }
    std::vector<double> y;
    pImpl->processWaveform(x.size(), samplingPeriod, x.data(), &y);
    return y;
}

/// Get the start sampling period 
double ProcessData::getTargetSamplingPeriod() const noexcept
{
    return pImpl->getTargetSamplingPeriod();
}

/// Create class
void PUUSSMLModels::OneComponentPicker::ZCNN::initializeProcessing(
    pybind11::module &m)
{
    pybind11::class_<ProcessData> p(m, "ProcessData");
    p.def(pybind11::init<> ());
    p.doc() = "Performs the preprocessing to use Zach Ross's fully connected neural network pick regressor architecture on UUSS data.\n\nProperties:\n\ntarget_sampling_period is the sampling period of the processed waveform in seconds.";

    p.def("process_waveform",
          &ProcessData::processWaveform,
          "Performs the appropriate preprocessing to the waveform with the given sampling period in seconds.");
    p.def_property_readonly("target_sampling_period",
                            &ProcessData::getTargetSamplingPeriod,
                            "The sampling period of the processed waveform in seconds.");
}
