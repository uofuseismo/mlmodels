#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include "uussmlmodels/pickers/cnnOneComponentP/inference.hpp"
#include "uussmlmodels/pickers/cnnOneComponentP/preprocessing.hpp"
#include "uussmlmodels/pickers/cnnThreeComponentS/inference.hpp"
#include "uussmlmodels/pickers/cnnThreeComponentS/preprocessing.hpp"
#include "pickers.hpp"
#include "buffer.hpp"

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
    auto xWork = ::bufferToVector<double>(x.request());
    if (samplingRate <= 0)
    {   
        throw std::invalid_argument("Sampling rate must be positive");
    }
    auto xProcessed = pImpl->process(xWork, samplingRate);
    return ::vectorToBuffer<double>(xProcessed);
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
    auto xWork = ::bufferToVector<double>(x.request());
    auto nSamples = static_cast<int> (xWork.size());
    if (nSamples != getExpectedSignalLength())
    {   
        throw std::invalid_argument("Signal must be length = "
                                  + std::to_string(getExpectedSignalLength()));
    }   
    return pImpl->predict(xWork);
}

///--------------------------------------------------------------------------///
///                               Three Component S                          ///
///--------------------------------------------------------------------------///

CNNThreeComponentS::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::Pickers::CNNThreeComponentS::Preprocessing> ()) 
{
}

double CNNThreeComponentS::Preprocessing::getTargetSamplingPeriod() const noexcept
{
    return UUSSMLModels::Pickers::CNNThreeComponentS::Preprocessing::getTargetSamplingPeriod();
}

double CNNThreeComponentS::Preprocessing::getTargetSamplingRate() const noexcept
{
    return UUSSMLModels::Pickers::CNNThreeComponentS::Preprocessing::getTargetSamplingRate();
}

void CNNThreeComponentS::Preprocessing::clear() noexcept
{
    pImpl->clear();
}

CNNThreeComponentS::Preprocessing::~Preprocessing() = default;

std::tuple<
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> >
CNNThreeComponentS::Preprocessing::process(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
    const double samplingRate)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling rate must be positive");
    }
    auto verticalWork = ::bufferToVector<double>(vertical.request());
    auto northWork    = ::bufferToVector<double>(north.request());
    auto eastWork     = bufferToVector<double>(east.request());
    auto result
        = pImpl->process(verticalWork, northWork, eastWork, samplingRate);
    return std::tuple {::vectorToBuffer<double>(std::get<0> (result)),
                       ::vectorToBuffer<double>(std::get<1> (result)),
                       ::vectorToBuffer<double>(std::get<2> (result))};
}


CNNThreeComponentS::Inference::Inference() :
    pImpl(std::make_unique<UUSSMLModels::Pickers::CNNThreeComponentS::Inference> ())
{
}

///--------------------------------------------------------------------------///
///                                Initialization                            ///
///--------------------------------------------------------------------------///
void UUSSMLModels::Python::Pickers::initialize(pybind11::module &m)
{
    pybind11::module pickersModule = m.def_submodule("Pickers");
    pickersModule.attr("__doc__") = "Phase pick regressors for refining preliminary P or S picks.";
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
R""""(
Preprocesses the single-channel vertical waveform.

Parameters
----------
signal : np.array
   The vertical-channel signal to preprocess.
sampling_rate : float
   The sampling rate of the signal in Hz.  This is assumed to be 100 Hz.
)"""",
        pybind11::arg("signal"),
        pybind11::arg("sampling_rate") = 100);

#if defined(WITH_TORCH) || defined(WITH_OPENVINO)
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
#endif

    ///----------------------------------------------------------------------///
    ///                             One Component S                          ///
    ///----------------------------------------------------------------------///
    pybind11::module threeComponentSModule = pickersModule.def_submodule("CNNThreeComponentS");
    /*
    pybind11::enum_<UUSSMLModels::Pickers::CNNThreeComponentS::Inference::Device> 
        (threeComponentSModule, "Device")
        .value("CPU", UUSSMLModels::Pickers::CNNThreeComponentS::Inference::Device::CPU,
               "Perform the inference on the CPU.")
        .value("GPU", UUSSMLModels::Pickers::CNNThreeComponentS::Inference::Device::GPU,
               "Perform the inference on the GPU.");
    */
    pybind11::enum_<UUSSMLModels::Pickers::CNNThreeComponentS::Inference::ModelFormat>
        (threeComponentSModule, "ModelFormat")
        .value("ONNX", UUSSMLModels::Pickers::CNNThreeComponentS::Inference::ModelFormat::ONNX,
               "The model is specified in ONNX format.")
        .value("HDF5", UUSSMLModels::Pickers::CNNThreeComponentS::Inference::ModelFormat::HDF5,
               "The model is specified in HDF5 format.");
        ;
    threeComponentSModule.attr("__doc__") = "S-phase pick regressor to be run on three-channel signals.";

    pybind11::class_<UUSSMLModels::Python::Pickers::CNNThreeComponentS::Preprocessing>
        threeComponentSPreprocessing(threeComponentSModule, "Preprocessing");
    threeComponentSPreprocessing.def(pybind11::init<> ());
    threeComponentSPreprocessing.doc() = R""""(
The preprocessing class for the three-component S pick regressor.

Read-Only Properties
    target_sampling_period : double
        The sampling period in seconds of the output signal.
    target_sampling_rate : double
        The sampling rate in Hz of the the output signal.
)"""";
    threeComponentSPreprocessing.def_property_readonly(
        "target_sampling_period",
        &CNNThreeComponentS::Preprocessing::getTargetSamplingPeriod);
    threeComponentSPreprocessing.def_property_readonly(
        "target_sampling_rate",
        &CNNThreeComponentS::Preprocessing::getTargetSamplingRate);
    threeComponentSPreprocessing.def(
        "clear",
        &CNNThreeComponentS::Preprocessing::clear,
        "Releases memory and resets the class.");
    threeComponentSPreprocessing.def(
        "process",
        &CNNThreeComponentS::Preprocessing::process,
R""""(
Preprocesses the three-component waveform.  Note, all signals must be the
same length.

Parameters
----------
vertical_signal : np.array
   The vertical signal to preprocess.
north_signal : np.array
   The north signal to preprocess.
east_signal : np.array
   The east signal to preprocess.
sampling_rate : float
   The sampling rate in Hz.  This is assumed to be 100 Hz.

Returns
-------
A tuple where the first item is the processed first (vertical) input signal,
the second item is the processed second (north) input signal, and the
third item is the processed third (east) input signal.
)"""",
        pybind11::arg("vertical_signal"),
        pybind11::arg("north_signal"),
        pybind11::arg("east_signal"),
        pybind11::arg("sampling_rate") = 100);
}
