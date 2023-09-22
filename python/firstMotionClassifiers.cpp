#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/preprocessing.hpp"
#include "firstMotionClassifiers.hpp"

using namespace UUSSMLModels::Python::FirstMotionClassifiers;

///--------------------------------------------------------------------------///
///                               One Component P                            ///
///--------------------------------------------------------------------------///

CNNOneComponentP::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Preprocessing> ())
{
}

double CNNOneComponentP::Preprocessing::getTargetSamplingPeriod() const noexcept
{
    return UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Preprocessing::getTargetSamplingPeriod();
}

double CNNOneComponentP::Preprocessing::getTargetSamplingRate() const noexcept
{
    return UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Preprocessing::getTargetSamplingRate();
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
    pImpl(std::make_unique<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference> ())
{
}

CNNOneComponentP::Inference::~Inference() = default;

void CNNOneComponentP::Inference::load(
    const std::string &fileName,
    const UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat format)
{
    pImpl->load(fileName, format);
}

bool CNNOneComponentP::Inference::isInitialized() const noexcept
{
    return pImpl->isInitialized();
}

double CNNOneComponentP::Inference::getSamplingRate() const noexcept
{
    return UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::getSamplingRate();
}

int CNNOneComponentP::Inference::getExpectedSignalLength() const noexcept
{
    return UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::getExpectedSignalLength();
}

double CNNOneComponentP::Inference::predict(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x)
{
    if (!isInitialized()){throw std::runtime_error("First motion classifier not initialized");}
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
void UUSSMLModels::Python::FirstMotionClassifiers::initialize(pybind11::module &m)
{
    pybind11::module fmModule = m.def_submodule("FirstMotionClassifiers");
    fmModule.attr("__doc__") = "P pick classifiers that assign P picks to up/down/unknown polarity.";
    ///----------------------------------------------------------------------///
    ///                             One Component P                          ///
    ///----------------------------------------------------------------------///
    pybind11::module oneComponentPModule = fmModule.def_submodule("CNNOneComponentP");
    /*
    pybind11::enum_<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::Device> 
        (oneComponentPModule, "Device")
        .value("CPU", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::Device::CPU,
               "Perform the inference on the CPU.")
        .value("GPU", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::Device::GPU,
               "Perform the inference on the GPU.");
    */
    pybind11::enum_<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat>
        (oneComponentPModule, "ModelFormat")
        .value("ONNX", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat::ONNX,
               "The model is specified in ONNX format.")
        .value("HDF5", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat::HDF5,
               "The model is specified in HDF5 format.");
        ;
    oneComponentPModule.attr("__doc__") = "P-phase pick regressor to be run on single (vertical) channel stations.";

    pybind11::class_<UUSSMLModels::Python::FirstMotionClassifiers::CNNOneComponentP::Preprocessing>
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

#if defined(WITH_TORCH) || defined(WITH_OPENVINO)
    pybind11::class_<UUSSMLModels::Python::FirstMotionClassifiers::CNNOneComponentP::Inference>
       oneComponentPInference(oneComponentPModule, "Inference");
    oneComponentPInference.def(pybind11::init<> ());
    oneComponentPInference.doc() = R""""(
The processing class for the one-component P-pick classifier.

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
    oneComponentPInference.def(
        "load",
        &CNNOneComponentP::Inference::load,
        "Loads the weights from file.",
        pybind11::arg("file_name"),
        pybind11::arg("model_format") = UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::ModelFormat::ONNX);
    oneComponentPInference.def(
        "predict",
        &CNNOneComponentP::Inference::predict,
        "Predicts the signal as up/down/unknown.",
        pybind11::arg("signal"));
#endif

}
