#include <uuss/threeComponentPicker/zrunet/processData.hpp>
#include <uuss/threeComponentPicker/zcnn/processData.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "processData.hpp"

using namespace PUUSSMLModels::ThreeComponentPicker;

///--------------------------------------------------------------------------///
///                                CNN Regressor                             ///
///--------------------------------------------------------------------------///
/// C'tor
ZCNN::ProcessData::ProcessData() :
    pImpl(std::make_unique<UUSS::ThreeComponentPicker::ZCNN::ProcessData> ())
{
}

/// Destructor
ZCNN::ProcessData::~ProcessData() = default;

std::vector<double> 
ZCNN::ProcessData::processWaveform(const std::vector<double> &x, 
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

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
ZCNN::ProcessData::processWaveforms3C(const std::vector<double> &z, 
                                      const std::vector<double> &n, 
                                      const std::vector<double> &e,
                                      const double samplingPeriod)
{
    auto t = std::tuple<const std::vector<double> &,
                        const std::vector<double> &,
                        const std::vector<double> &> (z, n, e); 
    return pImpl->processWaveforms(t, samplingPeriod);
}


/// Get the start sampling period 
double ZCNN::ProcessData::getTargetSamplingPeriod() const noexcept
{
    return pImpl->getTargetSamplingPeriod();
}

/// Create class
void PUUSSMLModels::ThreeComponentPicker::ZCNN::initializeProcessing(
    pybind11::module &m)
{
    pybind11::class_<ZCNN::ProcessData> p(m, "ProcessData");
    p.def(pybind11::init<> ());
    p.doc() = "Performs the preprocessing to use Zach Ross's fully connected neural network pick regressor architecture on UUSS data.\n\nProperties:\n\ntarget_sampling_period is the sampling period of the processed waveform in seconds.";

    p.def("process_waveform",
          &ProcessData::processWaveform,
          "Performs the appropriate preprocessing to the waveform with the given sampling period in seconds.");
    p.def("process_three_component_waveform",
          &ProcessData::processWaveforms3C,
          "Processes a three-component waveform.  The input waveforms are ordered (vertical, north, east).  The resulting waveforms are the processed (vertical, north, east) waveforms."); 
    p.def_property_readonly("target_sampling_period",
                            &ProcessData::getTargetSamplingPeriod,
                            "The sampling period of the processed waveform in seconds.");
}

///--------------------------------------------------------------------------///
///                                 UNet                                     ///
///--------------------------------------------------------------------------///
/// C'tor
ZRUNet::ProcessData::ProcessData() :
    pImpl(std::make_unique<UUSS::ThreeComponentPicker::ZRUNet::ProcessData> ()) 
{
}

/// Destructor
ZRUNet::ProcessData::~ProcessData() = default;

/*
/// Process a 1C waveform
std::vector<double>
ZRUNet::ProcessData::processWaveform(const std::vector<double> &x,
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
*/

std::vector<double>
ZRUNet::ProcessData::processWaveform(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x,
    const double samplingPeriod)
{
    if (samplingPeriod <= 0)
    {
        throw std::invalid_argument("Sampling rate = "
                                  + std::to_string(samplingPeriod)
                                  + " must be positive");
    }
    std::vector<double> xWork(x.size());
    std::memcpy(xWork.data(), x.data(), xWork.size()*sizeof(double));
    std::vector<double> y;
    pImpl->processWaveform(xWork.size(), samplingPeriod, xWork.data(), &y);
    return y;
}

/*
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
ZRUNet::ProcessData::processWaveforms3C(const std::vector<double> &z,
                                        const std::vector<double> &n,
                                        const std::vector<double> &e,
                                        const double samplingPeriod)
{
    auto t = std::tuple<const std::vector<double> &,
                        const std::vector<double> &,
                        const std::vector<double> &> (z, n, e); 
    return pImpl->processWaveforms(t, samplingPeriod); 
}
*/
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
ZRUNet::ProcessData::processWaveforms3C(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &z,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &n,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &e,
    const double samplingPeriod)
{
    if (samplingPeriod <= 0)
    {
        throw std::invalid_argument("Sampling rate = "
                                  + std::to_string(samplingPeriod)
                                  + " must be positive");
    }
    std::vector<double> zWork(z.size());
    std::memcpy(zWork.data(), z.data(), zWork.size()*sizeof(double));

    std::vector<double> nWork(n.size());
    std::memcpy(nWork.data(), n.data(), nWork.size()*sizeof(double));

    std::vector<double> eWork(e.size());
    std::memcpy(eWork.data(), e.data(), eWork.size()*sizeof(double));

    auto t = std::tuple<const std::vector<double> &,
                        const std::vector<double> &,
                        const std::vector<double> &> (zWork, nWork, eWork); 
    return pImpl->processWaveforms(t, samplingPeriod); 
}

/// Get the start sampling period 
double ZRUNet::ProcessData::getTargetSamplingPeriod() const noexcept
{
    return pImpl->getTargetSamplingPeriod();
}


/// Create class
void PUUSSMLModels::ThreeComponentPicker::ZRUNet::initializeProcessing(
    pybind11::module &m) 
{
    pybind11::class_<ZRUNet::ProcessData> p(m, "ProcessData");
    p.def(pybind11::init<> ());
    p.doc() = "Performs the preprocessing to use the U-Net detector architecture for three-component waveforms.\n\nProperties:\n\ntarget_sampling_period is the sampling period of the processed waveform in seconds.";

    p.def("process_waveform",
          &ProcessData::processWaveform,
          "Performs the appropriate preprocessing to the waveform with the given sampling period in seconds.");
    p.def("process_three_component_waveform",
          &ProcessData::processWaveforms3C,
          "Processes a three-component waveform.  The input waveforms are ordered (vertical, north, east).  The resulting waveforms are the processed (vertical, north, east) waveforms."); 
    p.def_property_readonly("target_sampling_period",
                            &ProcessData::getTargetSamplingPeriod,
                            "The sampling period of the processed waveform in seconds.");
}


