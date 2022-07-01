#include <uuss/features/version.hpp>
#include "magnitude/initialize.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(pyuussFeatures, m)
{
    m.attr("__version__") = UUSS_FEATURES_VERSION;
    m.attr("__name__") = "pyuussFeatures,";
    m.attr("__doc__") = "A Python interface to the Univeristy of Utah Seismgoraph Stations feature extraction tool for machine learning models.";
    // Magnitude
    pybind11::module magnitudeModule = m.def_submodule("Magnitude");
    magnitudeModule.attr("__doc__") = "Tools for extracting features from which station magnitudes are computed.";
    PFeatures::Magnitude::initialize(magnitudeModule);
}
