#include "uuss/version.hpp"
#include "oneComponentPicker/model.hpp"
#include "oneComponentPicker/processData.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(pyuussmlmodels, modules)
{
    modules.attr("__version__") = UUSSMLMODELS_VERSION; 
    modules.attr("__doc__") = "A Python interface for applying UUSS ML models.";

    // One component picker
    auto oneComponentPicker = modules.def_submodule("OneComponentPicker");
    oneComponentPicker.attr("__doc__") = "Modules for detection and picking on the vertical channel seismogram.";

    auto oneComponentPickerZCNN = oneComponentPicker.def_submodule("ZCNN");
    oneComponentPickerZCNN.attr("__doc__") = "Evalutes the arrival time regressor described in P Wave Arrival Picking and First‐Motion Polarity Determination With Deep Learning.";
    PUUSSMLModels::OneComponentPicker::ZCNN::initializeProcessing(oneComponentPickerZCNN);
 
    // First motion
    auto firstMotion = modules.def_submodule("FirstMotion");
    firstMotion.attr("__doc__") = "Modules for estimating first motions.";
    
    auto firstMotionZCNN = firstMotion.def_submodule("ZCNN");
    firstMotionZCNN.attr("__doc__") = "Evalutes the first motion classifier described in P Wave Arrival Picking and First‐Motion Polarity Determination With Deep Learning.";
}