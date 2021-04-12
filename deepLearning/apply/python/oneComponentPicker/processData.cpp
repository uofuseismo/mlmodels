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

/// 
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

    p.def_property_readonly("target_sampling_period",
                            &ProcessData::getTargetSamplingPeriod,
                            "The sampling period of the processed waveform in seconds.");
}
