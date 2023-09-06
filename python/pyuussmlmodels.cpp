#include "uussmlmodels/version.hpp"
#include "detectors.hpp"
#include "pickers.hpp"
#include "eventClassifiers.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(pyuussmlmodels, m)
{
    m.attr("__version__") = UUSSMLMODELS_VERSION;
    m.attr("__name__") = "pyuussmlmodels";
    m.attr("__doc__") = "A Python interface to the Univeristy of Utah Seismograph Stations production machine learning models.";

    UUSSMLModels::Python::Detectors::initialize(m);
    UUSSMLModels::Python::Pickers::initialize(m);
}

