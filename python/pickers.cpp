#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include "uussmlmodels/pickers/cnnOneComponentP/inference.hpp"
#include "uussmlmodels/pickers/cnnOneComponentP/preprocessing.hpp"
#include "pickers.hpp"

using namespace UUSSMLModels::Python::Pickers;

///--------------------------------------------------------------------------///
///                               One Component P                            ///
///--------------------------------------------------------------------------///

CNNOneComponentP::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::Pickers::CNNOneComponentP::Preprocessing> ())
{
}

double CNNOneComponentP::Preprocessing::getTargetSamplingPeriod() const noexcept
{
    return UUSSMLModels::Pickers::CNNOneComponentP::Preprocessing::getTargetSamplingPeriod();
}

double CNNOneComponentP::Preprocessing::getTargetSamplingRate() const noexcept
{
    return UUSSMLModels::Pickers::CNNOneComponentP::Preprocessing::getTargetSamplingRate();
}

pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
CNNOneComponentP::Preprocessing::process(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x,
    const double samplingRate)
{
    pybind11::buffer_info xBuffer = x.request();
    auto nSamples = static_cast<int> (xBuffer.size);
    const double *xPointer = (double *) (xBuffer.ptr);
    if (xPointer == nullptr)
    {
        throw std::invalid_argument("x is null");
    }
    std::vector<double> xWork(nSamples);
    std::copy(xPointer, xPointer + nSamples, xWork.data());
    auto xProcessed = pImpl->process(xWork, samplingRate);
    pybind11::array_t<double, pybind11::array::c_style> y(xProcessed.size());
    pybind11::buffer_info yBuffer = y.request();
    auto yPointer = static_cast<double *> (yBuffer.ptr);
    std::copy(xProcessed.begin(), xProcessed.end(), yPointer);
    return y;
}

void CNNOneComponentP::Preprocessing::clear() noexcept
{
    pImpl->clear(); 
}

CNNOneComponentP::Preprocessing::~Preprocessing() = default;


CNNOneComponentP::Inference::Inference() :
    pImpl(std::make_unique<UUSSMLModels::Pickers::CNNOneComponentP::Inference> ())
{
}

CNNOneComponentP::Inference::~Inference() = default;

void CNNOneComponentP::Inference::load(
    const std::string &fileName,
    const UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat format)
{
    pImpl->load(fileName, format);
}

bool CNNOneComponentP::Inference::isInitialized() const noexcept
{
    return pImpl->isInitialized();
}

std::pair<double, double>
CNNOneComponentP::Inference::getMinimumAndMaximumPerturbation() const noexcept
{
    return UUSSMLModels::Pickers::CNNOneComponentP::Inference::getMinimumAndMaximumPerturbation();
}

double CNNOneComponentP::Inference::getSamplingRate() const noexcept
{
    return UUSSMLModels::Pickers::CNNOneComponentP::Inference::getSamplingRate();
}

int CNNOneComponentP::Inference::getExpectedSignalLength() const noexcept
{
    return UUSSMLModels::Pickers::CNNOneComponentP::Inference::getExpectedSignalLength();
}

double CNNOneComponentP::Inference::predict(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x)
{
    if (!isInitialized()){throw std::runtime_error("Picker not initialized");}
    pybind11::buffer_info xBuffer = x.request();
    auto nSamples = static_cast<int> (xBuffer.size);
    if (nSamples != getExpectedSignalLength())
    {
        throw std::invalid_argument("Signal must be length = "
                                  + std::to_string(getExpectedSignalLength()));
    }
    const double *xPointer = (double *) (xBuffer.ptr);
    if (xPointer == nullptr)
    {
        throw std::invalid_argument("x is null");
    }
    std::vector<double> xWork(nSamples);
    std::copy(xPointer, xPointer + nSamples, xWork.data());
    return pImpl->predict(xWork);
}

///--------------------------------------------------------------------------///
///                                Initialization                            ///
///--------------------------------------------------------------------------///
void UUSSMLModels::Python::Pickers::initialize(pybind11::module &m)
{
    pybind11::module pickersModule = m.def_submodule("Pickers");
    pickersModule.attr("__doc__") = "Phase pick regressors for cleanign up preliminary P or S picks.";
    ///----------------------------------------------------------------------///
    ///                             One Component P                          ///
    ///----------------------------------------------------------------------///
    pybind11::module oneComponentPModule = pickersModule.def_submodule("CNNOneComponentP");
    /*
    pybind11::enum_<UUSSMLModels::Pickers::CNNOneComponentP::Inference::Device> 
        (oneComponentPModule, "Device")
        .value("CPU", UUSSMLModels::Pickers::CNNOneComponentP::Inference::Device::CPU,
               "Perform the inference on the CPU.")
        .value("GPU", UUSSMLModels::Pickers::CNNOneComponentP::Inference::Device::GPU,
               "Perform the inference on the GPU.");
    */
    pybind11::enum_<UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat>
        (oneComponentPModule, "ModelFormat")
        .value("ONNX", UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat::ONNX,
               "The model is specified in ONNX format.")
        .value("HDF5", UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat::HDF5,
               "The model is specified in HDF5 format.");
        ;
    oneComponentPModule.attr("__doc__") = "P-phase pick regressor to be run on single (vertical) channel stations.";

    pybind11::class_<UUSSMLModels::Python::Pickers::CNNOneComponentP::Preprocessing>
        oneComponentPPreprocessing(oneComponentPModule, "Preprocessing");
    oneComponentPPreprocessing.def(pybind11::init<> ());
    oneComponentPPreprocessing.doc() = R""""(
The preprocessing class for the one-component P pick regression.

Read-Only Properties
    target_sampling_period : double
        The sampling period in seconds of the output signal.
    target_sampling_rate : double
        The sampling rate in Hz of the the output signal.
)"""";
    oneComponentPPreprocessing.def_property_readonly(
        "target_sampling_period",
        &CNNOneComponentP::Preprocessing::getTargetSamplingPeriod);
    oneComponentPPreprocessing.def_property_readonly(
        "target_sampling_rate",
        &CNNOneComponentP::Preprocessing::getTargetSamplingRate);
    oneComponentPPreprocessing.def(
        "clear",
        &CNNOneComponentP::Preprocessing::clear,
        "Releases memory and resets the class.");
    oneComponentPPreprocessing.def(
        "process",
        &CNNOneComponentP::Preprocessing::process,
        "Preprocesses the waveform.",
        pybind11::arg("vertical_channel_signal"),
        pybind11::arg("samplingRate") = 100);

    pybind11::class_<UUSSMLModels::Python::Pickers::CNNOneComponentP::Inference>
       oneComponentPInference(oneComponentPModule, "Inference");
    oneComponentPInference.def(pybind11::init<> ());
    oneComponentPInference.doc() = R""""(
The processing class for the one-component P-pick regressor.

Read-Only Properties

    is_initialized : bool
        True indicates the class is initialized.
    sampling_rate : double
        The sampling rate of the input signal and output probability signal.
    expected_signal_length : int
        When not using the sliding window inference this is the expected signal length.
    minimum_and_maximum_pertrubation : double, double
        The minimum and maximum pick perturbation the model will produce measured in seconds.
)"""";
    oneComponentPInference.def_property_readonly(
        "is_initialized",
        &CNNOneComponentP::Inference::isInitialized);
    oneComponentPInference.def_property_readonly(
        "sampling_rate",
        &CNNOneComponentP::Inference::getSamplingRate);
    oneComponentPInference.def_property_readonly(
        "expected_signal_length",
        &CNNOneComponentP::Inference::getExpectedSignalLength);
    oneComponentPInference.def_property_readonly(
        "minimum_and_maximum_perturbation",
        &CNNOneComponentP::Inference::getMinimumAndMaximumPerturbation);
    oneComponentPInference.def(
        "load",
        &CNNOneComponentP::Inference::load,
        "Loads the weights from file.",
        pybind11::arg("file_name"),
        pybind11::arg("model_format") = UUSSMLModels::Pickers::CNNOneComponentP::Inference::ModelFormat::ONNX);
    oneComponentPInference.def(
        "predict",
        &CNNOneComponentP::Inference::predict,
        "Computes the perturbation, measured in seconds, to add to the initial pick.",
        pybind11::arg("signal"));


}
