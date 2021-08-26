#include "uuss/version.hpp"
#include "oneComponentPicker/model.hpp"
#include "oneComponentPicker/processData.hpp"
#include "threeComponentPicker/processData.hpp"
#include "firstMotion/processData.hpp"
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
    
    auto firstMotionFMNet = firstMotion.def_submodule("FMNet");
    firstMotionFMNet.attr("__doc__") = "Evalutes the first motion classifier described in P Wave Arrival Picking and First‐Motion Polarity Determination With Deep Learning.";
    PUUSSMLModels::FirstMotion::FMNet::initializeProcessing(firstMotionFMNet);

    // Three component picker
    auto threeComponentPicker = modules.def_submodule("ThreeComponentPicker");
    threeComponentPicker.attr("__doc__") = "Modules for detection and picking on three-component seismograms.";
    
    auto threeComponentPickerZCNN = threeComponentPicker.def_submodule("ZCNN");
    threeComponentPickerZCNN.attr("__doc__") = "Extends the arrival time regressor described in P Wave Arrival Picking and First-Motion Polarity Determination With Deep Learning to S waves.";
    PUUSSMLModels::ThreeComponentPicker::ZCNN::initializeProcessing(threeComponentPickerZCNN);

    auto threeComponentPickerZRUNet = threeComponentPicker.def_submodule("ZRUNet");
    threeComponentPickerZRUNet.attr("__doc__") = "Performs the pre-processing for the three-component UNet detector.";

}
