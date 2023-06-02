#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include "uussmlmodels/detectors/uNetOneComponentP/inference.hpp"
#include "uussmlmodels/detectors/uNetOneComponentP/preprocessing.hpp"
#include "detectors.hpp"

using namespace UUSSMLModels::Python::Detectors;

///--------------------------------------------------------------------------///
///                               One Component P                            ///
///--------------------------------------------------------------------------///

UNetOneComponentP::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::Detectors::UNetOneComponentP::Preprocessing> ())
{
}

double UNetOneComponentP::Preprocessing::getTargetSamplingPeriod() const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Preprocessing::getTargetSamplingPeriod();
}

double UNetOneComponentP::Preprocessing::getTargetSamplingRate() const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Preprocessing::getTargetSamplingRate();
}

pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
UNetOneComponentP::Preprocessing::process(
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

void UNetOneComponentP::Preprocessing::clear() noexcept
{
    pImpl->clear(); 
}

UNetOneComponentP::Preprocessing::~Preprocessing() = default;

///--------------------------------------------------------------------------///

/// Constructor
UNetOneComponentP::Inference::Inference() :
    pImpl(std::make_unique<UUSSMLModels::Detectors::UNetOneComponentP::Inference> ())
{
}

/// Destructor
UNetOneComponentP::Inference::~Inference() = default;

/// Process
void UNetOneComponentP::Inference::load(
    const std::string &fileName,
    const UUSSMLModels::Detectors::UNetOneComponentP::Inference::ModelFormat format)
{
    pImpl->load(fileName, format);
}

/// Initialized
bool UNetOneComponentP::Inference::isInitialized() const noexcept
{
    return pImpl->isInitialized();
}

double UNetOneComponentP::Inference::getSamplingRate() const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Inference::getSamplingRate();
}

int UNetOneComponentP::Inference::getMinimumSignalLength() const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Inference::getMinimumSignalLength();
}

int UNetOneComponentP::Inference::getExpectedSignalLength() const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Inference::getExpectedSignalLength();
}

bool UNetOneComponentP::Inference::isValidSignalLength(int nSamples) const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Inference::isValidSignalLength(nSamples);
}

std::pair<int, int> UNetOneComponentP::Inference::getCentralWindowStartEndIndex() const noexcept
{
    return UUSSMLModels::Detectors::UNetOneComponentP::Inference::getCentralWindowStartEndIndex();
}

/// Predict probability
pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> 
UNetOneComponentP::Inference::predictProbability(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x, 
    const bool useSlidingWindow)
{
    if (!isInitialized()){throw std::runtime_error("Detector not initialized");}
    pybind11::buffer_info xBuffer = x.request();
    auto nSamples = static_cast<int> (xBuffer.size);
    if (nSamples < getMinimumSignalLength())
    {
        throw std::invalid_argument("Signal must be length of at least "
                                  + std::to_string(getMinimumSignalLength()));
    }
    if (!useSlidingWindow)
    {
        if (isValidSignalLength(nSamples))
        {
            throw std::invalid_argument("Signal is not a valid length for non-sliding window");
        }
    }
    const double *xPointer = (double *) (xBuffer.ptr);
    if (xPointer == nullptr)
    {   
        throw std::invalid_argument("x is null");
    }   
    std::vector<double> xWork(nSamples);
    std::copy(xPointer, xPointer + nSamples, xWork.data());
    std::vector<double> probabilitySignal;
    if (useSlidingWindow)
    {
        probabilitySignal = pImpl->predictProbabilitySlidingWindow(xWork);
    }
    else
    {
        probabilitySignal = pImpl->predictProbability(xWork);
    }
    pybind11::array_t<double, pybind11::array::c_style>
        y(probabilitySignal.size());
    pybind11::buffer_info yBuffer = y.request();
    auto yPointer = static_cast<double *> (yBuffer.ptr);
    std::copy(probabilitySignal.begin(), probabilitySignal.end(), yPointer);
    return y;

}

///--------------------------------------------------------------------------///
///                                Initialization                            ///
///--------------------------------------------------------------------------///
void UUSSMLModels::Python::Detectors::initialize(pybind11::module &m)
{
    pybind11::module detectorsModule = m.def_submodule("Detectors");
    detectorsModule.attr("__doc__") = "Phase detectors for detecting P or S arrivals in continuous waveforms.";
    ///----------------------------------------------------------------------///
    ///                             One Component P                          ///
    ///----------------------------------------------------------------------///
    pybind11::module oneComponentPModule = detectorsModule.def_submodule("UNetOneComponentP");
    /*
    pybind11::enum_<UUSSMLModels::Detectors::UNetOneComponentP::Inference::Device> 
        (oneComponentPModule, "Device")
        .value("CPU", UUSSMLModels::Detectors::UNetOneComponentP::Inference::Device::CPU,
               "Perform the inference on the CPU.")
        .value("GPU", UUSSMLModels::Detectors::UNetOneComponentP::Inference::Device::GPU,
               "Perform the inference on the GPU.");
    */
    pybind11::enum_<UUSSMLModels::Detectors::UNetOneComponentP::Inference::ModelFormat> 
        (oneComponentPModule, "ModelFormat")
        .value("ONNX", UUSSMLModels::Detectors::UNetOneComponentP::Inference::ModelFormat::ONNX,
               "The model is specified in ONNX format.")
        //.value("HDF5", UUSSMLModels::Detectors::UNetOneComponentP::Inference::ModelFormat::HDF5,
        //       "The model is specified in HDF5 format.");
        ;
    oneComponentPModule.attr("__doc__") = "P-phase detectors to be run on single (vertical) channel stations."; 

    pybind11::class_<UUSSMLModels::Python::Detectors::UNetOneComponentP::Preprocessing>
        oneComponentPPreprocessing(oneComponentPModule, "Preprocessing");
    oneComponentPPreprocessing.def(pybind11::init<> ());
    oneComponentPPreprocessing.doc() = R""""(
The preprocessing class for the one-component P detector.

Read-Only Properties
    target_sampling_period : double
        The sampling period in seconds of the output signal.
    target_sampling_rate : double
        The sampling rate in Hz of the the output signal.
)"""";
    oneComponentPPreprocessing.def_property_readonly(
        "target_sampling_period",
        &UNetOneComponentP::Preprocessing::getTargetSamplingPeriod);
    oneComponentPPreprocessing.def_property_readonly(
        "target_sampling_rate",
        &UNetOneComponentP::Preprocessing::getTargetSamplingRate);
    oneComponentPPreprocessing.def(
        "clear",
        &UNetOneComponentP::Preprocessing::clear,
        "Releases memory and resets the class.");
    oneComponentPPreprocessing.def(
        "process",
        &UNetOneComponentP::Preprocessing::process,
        "Preprocesses the waveform.",
        pybind11::arg("vertical_channel_signal"),
        pybind11::arg("samplingRate") = 100);
        
    pybind11::class_<UUSSMLModels::Python::Detectors::UNetOneComponentP::Inference>
       oneComponentPInference(oneComponentPModule, "Inference");
    oneComponentPInference.def(pybind11::init<> ());
    oneComponentPInference.doc() = R""""(
The processing class for the one-component P detector.

Read-Only Properties

    is_initialized : bool
        True indicates the class is initialized.
    sampling_rate : double
        The sampling rate of the input signal and output probability signal.
    minimum_signal_length : int
        The minimum signal length required to apply the classifier.
    expected_signal_length : int
        When not using the sliding window inference this is the expected signal length.
    central_window_start_end_index : int, int
        An artifact of the training is that only the central portion of the
        window is valid.  This defines that window.  In the sliding window
        implementation the first n and last m samples are likely not valid.
 
)"""";
    oneComponentPInference.def_property_readonly(
        "is_initialized",
        &UNetOneComponentP::Inference::isInitialized);
    oneComponentPInference.def_property_readonly(
        "sampling_rate",
        &UNetOneComponentP::Inference::getSamplingRate);
    oneComponentPInference.def_property_readonly(
        "minimum_signal_length",
        &UNetOneComponentP::Inference::getMinimumSignalLength);
    oneComponentPInference.def_property_readonly(
        "expected_signal_length",
        &UNetOneComponentP::Inference::getExpectedSignalLength);
    oneComponentPInference.def_property_readonly(
        "central_window_start_end_index",
        &UNetOneComponentP::Inference::getCentralWindowStartEndIndex);
    oneComponentPInference.def(
        "valid_signal_length",
        &UNetOneComponentP::Inference::isValidSignalLength,
        "Determines whether or not the given signal length is valid for the non-sliding window processing.");
    oneComponentPInference.def(
        "load",
        &UNetOneComponentP::Inference::load,
        "Loads the weights from file.",
        pybind11::arg("file_name"),
        pybind11::arg("model_format") = UUSSMLModels::Detectors::UNetOneComponentP::Inference::ModelFormat::ONNX);
    oneComponentPInference.def(
        "predict_probability",
        &UNetOneComponentP::Inference::predictProbability,
        "Processes the given signal either in a sliding-window sense (default) or as a one-off signal",
        pybind11::arg("signal"),
        pybind11::arg("use_sliding_window") = true); 
}
