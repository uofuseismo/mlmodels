#include <iostream>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include "uussmlmodels/detectors/uNetOneComponentP/inference.hpp"
#include "uussmlmodels/detectors/uNetOneComponentP/preprocessing.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentP/inference.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentP/preprocessing.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentS/inference.hpp"
#include "uussmlmodels/detectors/uNetThreeComponentS/preprocessing.hpp"
#include "detectors.hpp"
#include "buffer.hpp"

using namespace UUSSMLModels::Python::Detectors;

/*
namespace
{
template<typename T>
std::vector<T> bufferToVector(const pybind11::buffer_info &buffer)
{
    auto nSamples = static_cast<int> (buffer.size);
    const T *pointer = (T *) (buffer.ptr);
    if (pointer == nullptr)
    {
        throw std::invalid_argument("Buffer data is null");
    }
    std::vector<T> work(nSamples);
    std::copy(pointer, pointer + nSamples, work.data());
    return work;
}
template<typename T>
pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>
   vectorToBuffer(const std::vector<T> &vector)
{
    pybind11::array_t<T, pybind11::array::c_style> buffer(vector.size());
    pybind11::buffer_info bufferHandle = buffer.request();
    auto pointer = static_cast<double *> (bufferHandle.ptr);
    std::copy(vector.begin(), vector.end(), pointer);
    return buffer;
}

}
*/

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
    auto xWork = ::bufferToVector<double>(x.request());
    auto xProcessed = pImpl->process(xWork, samplingRate);
    return ::vectorToBuffer(xProcessed);
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
    auto xWork = ::bufferToVector<double>(x.request());
    auto nSamples = static_cast<int> (xWork.size());
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
    std::vector<double> probabilitySignal;
    if (useSlidingWindow)
    {
        probabilitySignal = pImpl->predictProbabilitySlidingWindow(xWork);
    }
    else
    {
        probabilitySignal = pImpl->predictProbability(xWork);
    }
    return ::vectorToBuffer<double>(probabilitySignal);
}

///--------------------------------------------------------------------------///
///                             Three Component P                            ///
///--------------------------------------------------------------------------///

UNetThreeComponentP::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::Detectors::UNetThreeComponentP::Preprocessing> ())
{
}

double UNetThreeComponentP::Preprocessing::getTargetSamplingPeriod() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Preprocessing::getTargetSamplingPeriod();
}

double UNetThreeComponentP::Preprocessing::getTargetSamplingRate() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Preprocessing::getTargetSamplingRate();
}

std::tuple<
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> >
UNetThreeComponentP::Preprocessing::process(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
    const double samplingRate)
{
    auto vWork = ::bufferToVector<double>(vertical.request());
    auto nWork = ::bufferToVector<double>(north.request());
    auto eWork = ::bufferToVector<double>(east.request());
    if (vWork.size() != nWork.size() || vWork.size() != eWork.size())
    {
        throw std::invalid_argument("Inconsistent signal lengths");
    }
    auto processedSignals = pImpl->process(vWork, nWork, eWork, samplingRate);
    auto vProcessed = ::vectorToBuffer<double> (std::get<0>(processedSignals));
    auto nProcessed = ::vectorToBuffer<double> (std::get<1>(processedSignals));
    auto eProcessed = ::vectorToBuffer<double> (std::get<2>(processedSignals));
    return std::tuple {vProcessed, nProcessed, eProcessed};
}

void UNetThreeComponentP::Preprocessing::clear() noexcept
{
    pImpl->clear();
}

UNetThreeComponentP::Preprocessing::~Preprocessing() = default;

///--------------------------------------------------------------------------///

/// Constructor
UNetThreeComponentP::Inference::Inference() :
    pImpl(std::make_unique<UUSSMLModels::Detectors::UNetThreeComponentP::Inference> ())
{
}

/// Destructor
UNetThreeComponentP::Inference::~Inference() = default;

/// Process
void UNetThreeComponentP::Inference::load(
    const std::string &fileName,
    const UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ModelFormat format)
{
    pImpl->load(fileName, format);
}

/// Initialized
bool UNetThreeComponentP::Inference::isInitialized() const noexcept
{
    return pImpl->isInitialized();
}

double UNetThreeComponentP::Inference::getSamplingRate() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Inference::getSamplingRate();
}

int UNetThreeComponentP::Inference::getMinimumSignalLength() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Inference::getMinimumSignalLength();
}

int UNetThreeComponentP::Inference::getExpectedSignalLength() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Inference::getExpectedSignalLength();
}

bool UNetThreeComponentP::Inference::isValidSignalLength(int nSamples) const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Inference::isValidSignalLength(nSamples);
}

std::pair<int, int> UNetThreeComponentP::Inference::getCentralWindowStartEndIndex() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentP::Inference::getCentralWindowStartEndIndex();
}

/// Predict probability
pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> 
UNetThreeComponentP::Inference::predictProbability(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical, 
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
    const bool useSlidingWindow)
{
    if (!isInitialized()){throw std::runtime_error("Detector not initialized");}
    auto vWork = ::bufferToVector<double>(vertical.request());
    auto nWork = ::bufferToVector<double>(north.request());
    auto eWork = ::bufferToVector<double>(east.request());
    if (vWork.size() != nWork.size())
    {
        throw std::invalid_argument("Vertical signal length must equal north signal length");
    }
    if (vWork.size() != eWork.size())
    {
        throw std::invalid_argument("Vertical signal length must equal east signal length");
    }
    auto nSamples = static_cast<int> (vWork.size());
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
    std::vector<double> probabilitySignal;
    if (useSlidingWindow)
    {   
        probabilitySignal = pImpl->predictProbabilitySlidingWindow(vWork, nWork, eWork);
    }
    else
    {
        probabilitySignal = pImpl->predictProbability(vWork, nWork, eWork);
    }
    return ::vectorToBuffer<double>(probabilitySignal);
}

///--------------------------------------------------------------------------///
///                             Three Component S                            ///
///--------------------------------------------------------------------------///

UNetThreeComponentS::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::Detectors::UNetThreeComponentS::Preprocessing> ())
{
}

double UNetThreeComponentS::Preprocessing::getTargetSamplingPeriod() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Preprocessing::getTargetSamplingPeriod();
}

double UNetThreeComponentS::Preprocessing::getTargetSamplingRate() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Preprocessing::getTargetSamplingRate();
}

std::tuple<
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> >
UNetThreeComponentS::Preprocessing::process(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
    const double samplingRate)
{
    auto vWork = ::bufferToVector<double>(vertical.request());
    auto nWork = ::bufferToVector<double>(north.request());
    auto eWork = ::bufferToVector<double>(east.request());
    if (vWork.size() != nWork.size() || vWork.size() != eWork.size())
    {   
        throw std::invalid_argument("Inconsistent signal lengths");
    }   
    auto processedSignals = pImpl->process(vWork, nWork, eWork, samplingRate);
    auto vProcessed = ::vectorToBuffer<double> (std::get<0>(processedSignals));
    auto nProcessed = ::vectorToBuffer<double> (std::get<1>(processedSignals));
    auto eProcessed = ::vectorToBuffer<double> (std::get<2>(processedSignals));
    return std::tuple {vProcessed, nProcessed, eProcessed};
}

void UNetThreeComponentS::Preprocessing::clear() noexcept
{
    pImpl->clear();
}

UNetThreeComponentS::Preprocessing::~Preprocessing() = default;

///--------------------------------------------------------------------------///

/// Constructor
UNetThreeComponentS::Inference::Inference() :
    pImpl(std::make_unique<UUSSMLModels::Detectors::UNetThreeComponentS::Inference> ())
{
}

/// Destructor
UNetThreeComponentS::Inference::~Inference() = default;

/// Process
void UNetThreeComponentS::Inference::load(
    const std::string &fileName,
    const UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ModelFormat format)
{
    pImpl->load(fileName, format);
}

/// Initialized
bool UNetThreeComponentS::Inference::isInitialized() const noexcept
{
    return pImpl->isInitialized();
}

double UNetThreeComponentS::Inference::getSamplingRate() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Inference::getSamplingRate();
}

int UNetThreeComponentS::Inference::getMinimumSignalLength() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Inference::getMinimumSignalLength();
}

int UNetThreeComponentS::Inference::getExpectedSignalLength() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Inference::getExpectedSignalLength();
}

bool UNetThreeComponentS::Inference::isValidSignalLength(int nSamples) const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Inference::isValidSignalLength(nSamples);
}

std::pair<int, int> UNetThreeComponentS::Inference::getCentralWindowStartEndIndex() const noexcept
{
    return UUSSMLModels::Detectors::UNetThreeComponentS::Inference::getCentralWindowStartEndIndex();
}

/// Predict probability
pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
UNetThreeComponentS::Inference::predictProbability(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
    const bool useSlidingWindow)
{
    if (!isInitialized()){throw std::runtime_error("Detector not initialized");}
    auto vWork = ::bufferToVector<double>(vertical.request());
    auto nWork = ::bufferToVector<double>(north.request());
    auto eWork = ::bufferToVector<double>(east.request());
    if (vWork.size() != nWork.size())
    {
        throw std::invalid_argument("Vertical signal length must equal north signal length");
    }
    if (vWork.size() != eWork.size())
    {
        throw std::invalid_argument("Vertical signal length must equal east signal length");
    }
    auto nSamples = static_cast<int> (vWork.size());
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
    std::vector<double> probabilitySignal;
    if (useSlidingWindow)
    {
        probabilitySignal = pImpl->predictProbabilitySlidingWindow(vWork, nWork, eWork);
    }
    else
    {
        probabilitySignal = pImpl->predictProbability(vWork, nWork, eWork);
    }
    return ::vectorToBuffer<double>(probabilitySignal);
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
R""""(
Preprocesses the single-channel vertical waveform.

Parameters
----------
signal : np.array
   The vertical-channel signal to preprocess.
sampling_rate : float
   The sampling rate of the signal in Hz.  This is assumed to be 100 Hz.
)"""",
        "Preprocesses the waveform.",
        pybind11::arg("vertical_channel_signal"),
        pybind11::arg("sampling_rate") = 100);

#if defined(WITH_TORCH) || defined(WITH_OPENVINO)
    pybind11::class_<UUSSMLModels::Python::Detectors::UNetOneComponentP::Inference>
       oneComponentPInference(oneComponentPModule, "Inference");
    oneComponentPInference.def(pybind11::init<> ());
    oneComponentPInference.doc() = R""""(
The inference class for the one-component P detector.

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
R""""(
Performs inference on the preporcessed the one-component waveform.

Parameters
----------
vertical_signal : np.array
   The vertical signal on which to perform inference.
use_sliding_window : bool 
   If true, then this will apply a sliding window the central samples
   of the waveform (see central_window_start_end_index()).  Otherwise, this
   performs a one-off inference on the expected_signal_length() signal. 

Returns
-------
A probability signal with sampling rate given by sampling_rate().  When this
is near one then the sample is likely corresponding to a P wave.  Otherwise,
the sample likely corresponds to noise.
)"""",
        pybind11::arg("signal"),
        pybind11::arg("use_sliding_window") = true); 
#endif
    ///----------------------------------------------------------------------///
    ///                            Three Component P                         ///
    ///----------------------------------------------------------------------///
    pybind11::module threeComponentPModule = detectorsModule.def_submodule("UNetThreeComponentP");
    /*  
    pybind11::enum_<UUSSMLModels::Detectors::UNetThreeComponentP::Inference::Device> 
        (threeComponentPModule, "Device")
        .value("CPU", UUSSMLModels::Detectors::UNetThreeComponentP::Inference::Device::CPU,
               "Perform the inference on the CPU.")
        .value("GPU", UUSSMLModels::Detectors::UNetThreeComponentP::Inference::Device::GPU,
               "Perform the inference on the GPU.");
    */
    pybind11::enum_<UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ModelFormat> 
        (threeComponentPModule, "ModelFormat")
        .value("ONNX", UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ModelFormat::ONNX,
               "The model is specified in ONNX format.")
        //.value("HDF5", UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ModelFormat::HDF5,
        //       "The model is specified in HDF5 format.");
        ;
    threeComponentPModule.attr("__doc__") = "P-phase detectors to be run on three-component sensors.";

    pybind11::class_<UUSSMLModels::Python::Detectors::UNetThreeComponentP::Preprocessing>
        threeComponentPPreprocessing(threeComponentPModule, "Preprocessing");
    threeComponentPPreprocessing.def(pybind11::init<> ());
    threeComponentPPreprocessing.doc() = R""""(
The preprocessing class for the three-component P detector.

Read-Only Properties
    target_sampling_period : double
        The sampling period in seconds of the output signals.
    target_sampling_rate : double
        The sampling rate in Hz of the the output signal.
)"""";
    threeComponentPPreprocessing.def_property_readonly(
        "target_sampling_period",
        &UNetThreeComponentP::Preprocessing::getTargetSamplingPeriod);
    threeComponentPPreprocessing.def_property_readonly(
        "target_sampling_rate",
        &UNetThreeComponentP::Preprocessing::getTargetSamplingRate);
    threeComponentPPreprocessing.def(
        "clear",
        &UNetThreeComponentP::Preprocessing::clear,
        "Releases memory and resets the class.");
    threeComponentPPreprocessing.def(
        "process",
        &UNetThreeComponentP::Preprocessing::process,
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
#if defined(WITH_TORCH) || defined(WITH_OPENVINO)
    pybind11::class_<UUSSMLModels::Python::Detectors::UNetThreeComponentP::Inference>
       threeComponentPInference(threeComponentPModule, "Inference");
    threeComponentPInference.def(pybind11::init<> ());
    threeComponentPInference.doc() = R""""(
The inference class for the three-component P detector.

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
    threeComponentPInference.def_property_readonly(
        "is_initialized",
        &UNetThreeComponentP::Inference::isInitialized);
    threeComponentPInference.def_property_readonly(
        "sampling_rate",
        &UNetThreeComponentP::Inference::getSamplingRate);
    threeComponentPInference.def_property_readonly(
        "minimum_signal_length",
        &UNetThreeComponentP::Inference::getMinimumSignalLength);
    threeComponentPInference.def_property_readonly(
        "expected_signal_length",
        &UNetThreeComponentP::Inference::getExpectedSignalLength);
    threeComponentPInference.def_property_readonly(
        "central_window_start_end_index",
        &UNetThreeComponentP::Inference::getCentralWindowStartEndIndex);
    threeComponentPInference.def(
        "valid_signal_length",
        &UNetThreeComponentP::Inference::isValidSignalLength,
        "Determines whether or not the given signal length is valid for the non-sliding window processing.");
    threeComponentPInference.def(
        "load",
        &UNetThreeComponentP::Inference::load,
        "Loads the weights from file.",
        pybind11::arg("file_name"),
        pybind11::arg("model_format") = UUSSMLModels::Detectors::UNetThreeComponentP::Inference::ModelFormat::ONNX);
    threeComponentPInference.def(
        "predict_probability",
        &UNetThreeComponentP::Inference::predictProbability,
R""""(
Performs inference on the preporcessed the three-component waveform.  Note, all
signals must be the same length.

Parameters
----------
vertical_signal : np.array
   The vertical signal on which to perform inference.
north_signal : np.array
   The north signal on which to perform inference.
east_signal : np.array
   The east signal to preprocess.
use_sliding_window : bool 
   If true, then this will apply a sliding window the central samples
   of the waveform (see central_window_start_end_index()).  Otherwise, this
   performs a one-off inference on the expected_signal_length() signal. 

Returns
-------
A probability signal with sampling rate given by sampling_rate().  When this
is near one then the sample is likely corresponding to a P wave.  Otherwise,
the sample likely corresponds to noise.
)"""",
        pybind11::arg("vertical_signal"),
        pybind11::arg("north_signal"),
        pybind11::arg("east_signal"),
        pybind11::arg("use_sliding_window") = true);
#endif
    ///----------------------------------------------------------------------///
    ///                            Three Component S                         ///
    ///----------------------------------------------------------------------///
    pybind11::module threeComponentSModule = detectorsModule.def_submodule("UNetThreeComponentS");
    /*  
    pybind11::enum_<UUSSMLModels::Detectors::UNetThreeComponentS::Inference::Device> 
        (threeComponentSModule, "Device")
        .value("CPU", UUSSMLModels::Detectors::UNetThreeComponentS::Inference::Device::CPU,
               "Perform the inference on the CPU.")
        .value("GPU", UUSSMLModels::Detectors::UNetThreeComponentS::Inference::Device::GPU,
               "Perform the inference on the GPU.");
    */
    pybind11::enum_<UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ModelFormat>
        (threeComponentSModule, "ModelFormat")
        .value("ONNX", UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ModelFormat::ONNX,
               "The model is specified in ONNX format.")
        //.value("HDF5", UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ModelFormat::HDF5,
        //       "The model is specified in HDF5 format.");
        ;
    threeComponentSModule.attr("__doc__") = "S-phase detectors to be run on three-component sensors.";

    pybind11::class_<UUSSMLModels::Python::Detectors::UNetThreeComponentS::Preprocessing>
        threeComponentSPreprocessing(threeComponentSModule, "Preprocessing");
    threeComponentSPreprocessing.def(pybind11::init<> ());
    threeComponentSPreprocessing.doc() = R""""(
The preprocessing class for the three-component S detector.

Read-Only Properties
    target_sampling_period : double
        The sampling period in seconds of the output signals.
    target_sampling_rate : double
        The sampling rate in Hz of the the output signal.
)"""";
    threeComponentSPreprocessing.def_property_readonly(
        "target_sampling_period",
        &UNetThreeComponentS::Preprocessing::getTargetSamplingPeriod);
    threeComponentSPreprocessing.def_property_readonly(
        "target_sampling_rate",
        &UNetThreeComponentS::Preprocessing::getTargetSamplingRate);
    threeComponentSPreprocessing.def(
        "clear",
        &UNetThreeComponentS::Preprocessing::clear,
        "Releases memory and resets the class.");
    threeComponentSPreprocessing.def(
        "process",
        &UNetThreeComponentS::Preprocessing::process,
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
#if defined(WITH_TORCH) || defined(WITH_OPENVINO)
    pybind11::class_<UUSSMLModels::Python::Detectors::UNetThreeComponentS::Inference>
       threeComponentSInference(threeComponentSModule, "Inference");
    threeComponentSInference.def(pybind11::init<> ());
    threeComponentSInference.doc() = R""""(
The inference class for the three-component S detector.

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
    threeComponentSInference.def_property_readonly(
        "is_initialized",
        &UNetThreeComponentS::Inference::isInitialized);
    threeComponentSInference.def_property_readonly(
        "sampling_rate",
        &UNetThreeComponentS::Inference::getSamplingRate);
    threeComponentSInference.def_property_readonly(
        "minimum_signal_length",
        &UNetThreeComponentS::Inference::getMinimumSignalLength);
    threeComponentSInference.def_property_readonly(
        "expected_signal_length",
        &UNetThreeComponentS::Inference::getExpectedSignalLength);
    threeComponentSInference.def_property_readonly(
        "central_window_start_end_index",
        &UNetThreeComponentS::Inference::getCentralWindowStartEndIndex);
    threeComponentSInference.def(
        "valid_signal_length",
        &UNetThreeComponentS::Inference::isValidSignalLength,
        "Determines whether or not the given signal length is valid for the non-sliding window processing.");
    threeComponentSInference.def(
        "load",
        &UNetThreeComponentS::Inference::load,
        "Loads the weights from file.",
        pybind11::arg("file_name"),
        pybind11::arg("model_format") = UUSSMLModels::Detectors::UNetThreeComponentS::Inference::ModelFormat::ONNX);
    threeComponentSInference.def(
        "predict_probability",
        &UNetThreeComponentS::Inference::predictProbability,
R""""(
Performs inference on the preporcessed the three-component waveform.  Note, all
signals must be the same length.

Parameters
----------
vertical_signal : np.array
   The vertical signal on which to perform inference.
north_signal : np.array
   The north signal on which to perform inference.
east_signal : np.array
   The east signal to preprocess.
use_sliding_window : bool 
   If true, then this will apply a sliding window the central samples
   of the waveform (see central_window_start_end_index()).  Otherwise, this
   performs a one-off inference on the expected_signal_length() signal. 

Returns
-------
A probability signal with sampling rate given by sampling_rate().  When this
is near one then the sample is likely corresponding to an S wave.  Otherwise,
the sample likely corresponds to noise.
)"""",
        pybind11::arg("vertical_signal"),
        pybind11::arg("north_signal"),
        pybind11::arg("east_signal"),
        pybind11::arg("use_sliding_window") = true);
#endif
}
