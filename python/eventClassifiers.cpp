#include <pybind11/stl.h>
#include <uussmlmodels/eventClassifiers/cnnThreeComponent/preprocessing.hpp>
#include "eventClassifiers.hpp"
#include "buffer.hpp"

using namespace UUSSMLModels::Python::EventClassifiers;

namespace
{
std::vector<double> transpose(const int nColumns, const int nRows, const std::vector<double> &x)
{
    std::vector<double> y(x.size());
    for (int i = 0; i < nColumns; ++i)
    {
        for (int j = 0; j < nRows; ++j)
        {
            int ij = j*nColumns + i;
            int ji = i*nRows    + j;
            y[ji] = x[ij];
        }
    }
    return y;
}

}

CNNThreeComponent::Preprocessing::Preprocessing() :
    pImpl(std::make_unique<UUSSMLModels::EventClassifiers::CNNThreeComponent::Preprocessing> ())
{
}

/*
double CNNThreeComponent::Preprocessing::getScalogramSamplingRate() const noexcept
{
    return pImpl->getScalogramSamplingRate();
}

double CNNThreeComponent::Preprocessing::getScalogramSamplingPeriod() const noexcept
{
    return pImpl->getScalogramSamplingPeriod();
}
*/

void CNNThreeComponent::Preprocessing::clear() noexcept
{
    pImpl->clear();
}

pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
CNNThreeComponent::Preprocessing::processVerticalChannel(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical,
    const double samplingRate)
{
    if (samplingRate <= 0)
    {   
        throw std::invalid_argument("Sampling rate must be positive");
    }   
    auto nFrequencies = pImpl->getNumberOfFrequencies();
    auto nSamples     = pImpl->getNumberOfTimeWindows();
    auto vWork = ::bufferToVector<double> (vertical.request());
    auto vSpectrogram = pImpl->process(vWork, samplingRate);
    auto vOut = ::vectorToBuffer<double> (vSpectrogram);
    vOut = vOut.reshape( {nSamples, nFrequencies} );
/*
    auto nScales  = pImpl->getNumberOfScales();
    auto nSamples = pImpl->getScalogramLength();
    auto vWork = ::bufferToVector<double> (vertical.request());
    auto vScalogram = pImpl->process(vWork, samplingRate);
    vScalogram = ::transpose(nSamples, nScales, vScalogram);
    auto vOut = ::vectorToBuffer<double> (vScalogram);
    vOut = vOut.reshape( {nSamples, nScales} );
*/
    return vOut;
}

std::vector<double> CNNThreeComponent::Preprocessing::getFrequencies() const
{
    return pImpl->getFrequencies();
}

int CNNThreeComponent::Preprocessing::getNumberOfFrequencies() const noexcept
{
    return pImpl->getNumberOfFrequencies();
}

std::vector<double> CNNThreeComponent::Preprocessing::getTimeWindows() const
{
    return pImpl->getTimeWindows();
}

int CNNThreeComponent::Preprocessing::getNumberOfTimeWindows() const noexcept
{
    return pImpl->getNumberOfTimeWindows();
}

/*
int CNNThreeComponent::Preprocessing::getNumberOfScales() const noexcept
{
    return pImpl->getNumberOfScales();
}

int CNNThreeComponent::Preprocessing::getScalogramLength() const noexcept
{
    return pImpl->getScalogramLength();
}
*/

std::tuple<
   pybind11::array_t<double>,//, pybind11::array::c_style | pybind11::array::forcecast>,
   pybind11::array_t<double>,//, pybind11::array::c_style | pybind11::array::forcecast>,
   pybind11::array_t<double>//, pybind11::array::c_style | pybind11::array::forcecast>
>
CNNThreeComponent::Preprocessing::process(
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &vertical, 
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &north,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &east,
    double samplingRate)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling rate must be positive");
    }
    auto nFrequencies = pImpl->getNumberOfFrequencies();
    auto nSamples     = pImpl->getNumberOfTimeWindows();
    auto vWork = ::bufferToVector<double> (vertical.request());
    auto nWork = ::bufferToVector<double> (north.request());
    auto eWork = ::bufferToVector<double> (east.request());
    auto [vSpectrogram, nSpectrogram, eSpectrogram]
        = pImpl->process(vWork, nWork, eWork, samplingRate);
    auto vOut = ::vectorToBuffer<double> (vSpectrogram);
    auto nOut = ::vectorToBuffer<double> (nSpectrogram);
    auto eOut = ::vectorToBuffer<double> (eSpectrogram);
    vOut = vOut.reshape( {nSamples, nFrequencies} );
    nOut = nOut.reshape( {nSamples, nFrequencies} );
    eOut = eOut.reshape( {nSamples, nFrequencies} );
/*
    auto nScales  = pImpl->getNumberOfScales();
    auto nSamples = pImpl->getScalogramLength();
    auto vWork = ::bufferToVector<double> (vertical.request());
    auto nWork = ::bufferToVector<double> (north.request());
    auto eWork = ::bufferToVector<double> (east.request());
    auto [vScalogram, nScalogram, eScalogram]
        = pImpl->process(vWork, nWork, eWork, samplingRate);
    vScalogram = ::transpose(nSamples, nScales, vScalogram);
    nScalogram = ::transpose(nSamples, nScales, nScalogram);
    eScalogram = ::transpose(nSamples, nScales, eScalogram);
    auto vOut = ::vectorToBuffer<double> (vScalogram);
    auto nOut = ::vectorToBuffer<double> (nScalogram);
    auto eOut = ::vectorToBuffer<double> (eScalogram);
    vOut = vOut.reshape( {nSamples, nScales} );
    nOut = nOut.reshape( {nSamples, nScales} );
    eOut = eOut.reshape( {nSamples, nScales} );
*/
    return std::tuple {vOut, nOut, eOut};
}
 
/*
pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
CNNThreeComponent::Preprocessing::getCenterFrequencies() const
{
    return ::vectorToBuffer<double>(pImpl->getCenterFrequencies());
}
*/

CNNThreeComponent::Preprocessing::~Preprocessing() = default;

///--------------------------------------------------------------------------///
///                                Initialization                            ///
///--------------------------------------------------------------------------///
void UUSSMLModels::Python::EventClassifiers::initialize(pybind11::module &m)
{
    pybind11::module ecModule = m.def_submodule("EventClassifiers");
    ecModule.attr("__doc__") = R""""(
Classifiers used for determing an event type.
)"""";
    ///----------------------------------------------------------------------///
    ///                            Three Component                           ///
    ///----------------------------------------------------------------------///
    pybind11::module threeComponentModule = ecModule.def_submodule("CNNThreeComponent");
    threeComponentModule.attr("__doc__") = "Classifies an event as an earthquake or quarry blast based on a waveform";

    pybind11::class_<UUSSMLModels::Python::EventClassifiers::CNNThreeComponent::Preprocessing>
        threeComponentPreprocessing(threeComponentModule, "Preprocessing");
    threeComponentPreprocessing.def(pybind11::init<> ());
    threeComponentPreprocessing.doc() = R""""(
The preprocessing class for the three-component wavform-based quarry blast/event classifier.

Read-Only Properties
    frequencies : np.array
        The frequencies of the short-time Fourier transform in Hz.
    number_of_frequencies : int
        The number of frequencies.
    number_of_time_windows : int
        The number of time windows.
    time_windows : np.array
        The start time of an short-time Fourier transform in seconds. 
)"""";
    threeComponentPreprocessing.def_property_readonly(
        "frequencies",
        &CNNThreeComponent::Preprocessing::getFrequencies);
    threeComponentPreprocessing.def_property_readonly(
        "number_of_frequencies",
        &CNNThreeComponent::Preprocessing::getNumberOfFrequencies);
    threeComponentPreprocessing.def_property_readonly(
        "number_of_time_windows",
        &CNNThreeComponent::Preprocessing::getNumberOfTimeWindows);
    threeComponentPreprocessing.def_property_readonly(
        "time_windows",
        &CNNThreeComponent::Preprocessing::getTimeWindows);

/*
    threeComponentPreprocessing.def_property_readonly(
        "center_frequencies",
        &CNNThreeComponent::Preprocessing::getCenterFrequencies);
    threeComponentPreprocessing.def_property_readonly(
        "number_of_scales",
        &CNNThreeComponent::Preprocessing::getNumberOfScales);
    threeComponentPreprocessing.def_property_readonly(
        "scalogram_length",
        &CNNThreeComponent::Preprocessing::getScalogramLength);
    threeComponentPreprocessing.def_property_readonly(
        "scalogram_sampling_period",
        &CNNThreeComponent::Preprocessing::getScalogramSamplingPeriod);
    threeComponentPreprocessing.def_property_readonly(
        "scalogram_sampling_rate",
        &CNNThreeComponent::Preprocessing::getScalogramSamplingRate);
*/
    threeComponentPreprocessing.def(
        "clear",
        &CNNThreeComponent::Preprocessing::clear,
        "Releases memory and resets the class.");
    threeComponentPreprocessing.def(
        "process",
        &CNNThreeComponent::Preprocessing::process,
R""""(
Preprocesses the threee-channel waveform.

Parameters
----------
vertical : np.array
   The vertical-channel signal to preprocess.
north: np.array
   The north-channel signal to preprocess.
east: np.array
   The east-channel signal to preprocess.
sampling_rate : float
   The sampling rate of the signal in Hz.  This is assumed to be 100 Hz.

Returns
-------
The corresponding vertical, north, and east spectrograms.  The first dimension
is the temporary dimension and the second dimension is the frequency dimension.
)"""",
        pybind11::arg("vertical"),
        pybind11::arg("north"),
        pybind11::arg("east"),
        pybind11::arg("sampling_rate") = 100);
    threeComponentPreprocessing.def(
        "process_vertical_channel",
        &CNNThreeComponent::Preprocessing::processVerticalChannel,
R""""(
Preprocesses the vertical channel waveform waveform.

Parameters
----------
vertical : np.array
   The vertical-channel signal to preprocess.
sampling_rate : float
   The sampling rate of the signal in Hz.  This is assumed to be 100 Hz.

Returns
-------
The corresponding vertical spectrogram.  The first dimension is the temporal
dimension and the second dimension is the frequency dimension.

Notes
-----
You can plot this by doing the following:

frequencies = preprocess.center_frequencies
time_windows = preprocess.time_windows
[v_spectrogram, n_spectrogram, e_spectrogram] = preprocess.process(vertical, north, east, 100.0)
# Transpose converts from matrix indexing to cartesian indexing for plotting.  The
# frequencies need to be reversed since matrices increase down and plots increase up 
plt.imshow(v_spectrogram[:,::-1].T, extent = [0, sampling_period*(v_spectorgram.shape[0] - 1), frequencies[0], frequencies[-1]])
)"""",
        pybind11::arg("vertical"),
        pybind11::arg("sampling_rate") = 100);

}
