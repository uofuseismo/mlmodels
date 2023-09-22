#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/preprocessing.hpp"
#include "firstMotionClassifiers.hpp"
#include "buffer.hpp"

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

void CNNOneComponentP::Inference::setProbabilityThreshold(const double threshold)
{
    if (threshold < 0 || threshold > 1)
    {
        throw std::invalid_argument("Threshold must be in range [0,1]");
    }
    pImpl->setProbabilityThreshold(threshold);
}

double CNNOneComponentP::Inference::getProbabilityThreshold() const noexcept
{
    return pImpl->getProbabilityThreshold();
}

double CNNOneComponentP::Inference::getSamplingRate() const noexcept
{
    return UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::getSamplingRate();
}

int CNNOneComponentP::Inference::getExpectedSignalLength() const noexcept
{
    return UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::getExpectedSignalLength();
}

UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::FirstMotion
CNNOneComponentP::Inference::predict(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x) const
{
    if (!isInitialized()){throw std::runtime_error("First motion classifier not initialized");}
    auto xWork = ::bufferToVector<double>(x.request());
    auto nSamples = static_cast<int> (xWork.size());
    if (nSamples != getExpectedSignalLength())
    {
        throw std::invalid_argument("Signal must be length = "
                                  + std::to_string(getExpectedSignalLength()));
    }
    return pImpl->predict(xWork);
}

std::tuple<double, double, double>
CNNOneComponentP::Inference::predictProbability(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x) const
{
    if (!isInitialized()){throw std::runtime_error("First motion classifier not initialized");}
    auto xWork = ::bufferToVector<double>(x.request());
    auto nSamples = static_cast<int> (xWork.size());
    if (nSamples != getExpectedSignalLength())
    {   
        throw std::invalid_argument("Signal must be length = "
                                  + std::to_string(getExpectedSignalLength()));
    }
    return pImpl->predictProbability<double>(xWork);
}

///--------------------------------------------------------------------------///
///                                Initialization                            ///
///--------------------------------------------------------------------------///
void UUSSMLModels::Python::FirstMotionClassifiers::initialize(pybind11::module &m)
{
    pybind11::module fmModule = m.def_submodule("FirstMotionClassifiers");
    fmModule.attr("__doc__") = R""""(
P pick classifiers that assign the first motion of arrival to an 
up, a down, or an unknown first motion.
)"""";
    ///----------------------------------------------------------------------///
    ///                             One Component P                          ///
    ///----------------------------------------------------------------------///
    pybind11::module oneComponentPModule = fmModule.def_submodule("CNNOneComponentP");
    pybind11::enum_<UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::FirstMotion> 
        (oneComponentPModule, "FirstMotion")
        .value("Up", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::FirstMotion::Up,
               "The first motion is up.")
        .value("Down", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::FirstMotion::Down,
               "The first motion is down.")
        .value("Unknown", UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::FirstMotion::Unknown,
               "The first motion is unknown - i.e., neither up or down.");
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
    pybind11::class_<UUSSMLModels::Python::FirstMotionClassifiers::CNNOneComponentP::Inference>
       oneComponentPInference(oneComponentPModule, "Inference");
    oneComponentPInference.def(pybind11::init<> ());
    oneComponentPInference.doc() = R""""(
The processing class for the one-component P-pick classifier.

Read-Write Properties
    probability_threshold : float
        The posterior probability that a first-motion pick must exceed to be 
        classified as an up or down pick.  For example, if this is 0.4 and the
        up probability is 0.42, the down probability is 0.41, and the
        unknown probability is 0.17 then this will classify as up since the up
        class exceeds the threshold and has the largest posterior probability.
        However, if this is 0.6 and the up probability is 0.5, the down
        probability, is 0.2, and the unknown probability is 0.4 then
        this will classify to unknown since the up probability did not exceed
        0.6.  By default we approximate the Bayes's classifier and classify to
        the class with the largest posterior probability - i.e., the default is
        1/3.
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
    oneComponentPInference.def_property(
       "probability_threshold",
        &CNNOneComponentP::Inference::getProbabilityThreshold,
        &CNNOneComponentP::Inference::setProbabilityThreshold);
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
R""""(
Predicts the signal as Up, Down, or Unknown.

Parameters
signal : np.array
    The signal whose first motion will be classified.  The pick should be
    centered half-way into the window.  Also, this signal's length must
    match the expected_signal_length.)

Returns
FirstMotion
   The first motion which will be Up, Down, or Unknown. 
)"""",
        pybind11::arg("signal"));
    oneComponentPInference.def(
        "predict_probability",
        &CNNOneComponentP::Inference::predictProbability,
R""""(
Predicts the posterior probability of the signal's first motion being
up, down, and unknown.

Parameters
signal : np.array
    The signal whose first motion will be classified.  The pick should be
    centered half-way into the window.  Also, this signal's length must
    match the expected_signal_length.)

Returns
A tuple where the first value is the probability of being up, the second
value is the probability of being down, and the third probabiliy the 
probability of being unknown.  These probabilities will sum to 1.
)"""",
        pybind11::arg("signal"));
#endif

}
