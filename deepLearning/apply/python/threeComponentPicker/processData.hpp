#ifndef PUUSSMLMODEL_THREECOMPONENTPICKER_PROCESSDATA_HPP
#define PUUSSMLMODEL_THREECOMPONENTPICKER_PROCESSDATA_HPP
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// Forward declarations
namespace UUSS::ThreeComponentPicker
{
    namespace ZCNN
    {
        class ProcessData;
    }
    namespace ZRUNet
    {
        class ProcessData;
    }
}
namespace PUUSSMLModels::ThreeComponentPicker
{

/// Three-component regressor pre-processing
namespace ZCNN
{
class ProcessData
{
public:
    /// C'tor
    ProcessData();
    /// Destructor
    ~ProcessData();
    /// Preprocess the three component signal.
    [[nodiscard]]
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
        processWaveforms3C(const std::vector<double> &z, 
                           const std::vector<double> &n, 
                           const std::vector<double> &e,
                           const double samplingPeriod = 0.01);
    /// Preprocess the input waveform.
/*
    [[nodiscard]]
    std::vector<double> processWaveform(
         py::array_t<double, py::array::c_style | py::array::forcecast> &x,
         const double samplingPeriod = 0.01);
*/
    [[nodiscard]]
    std::vector<double> processWaveform(const std::vector<double> &x,
                                        const double samplingPeriod = 0.01);
    /// @result The target sampling period in seconds.
    [[nodiscard]] double getTargetSamplingPeriod() const noexcept;

    ProcessData(const ProcessData &process) = delete;
    ProcessData(ProcessData &&process) noexcept = delete;
    ProcessData& operator=(const ProcessData &process) noexcept = delete;
    ProcessData& operator=(ProcessData &&process) = delete;
private:
    std::unique_ptr<UUSS::ThreeComponentPicker::ZCNN::ProcessData> pImpl;
};
void initializeProcessing(pybind11::module &m);
}

/// Three-component UNet pre-processing
namespace ZRUNet
{
class ProcessData
{
public:
    /// C'tor
    ProcessData();
    /// Destructor
    ~ProcessData();
    /// Preprocess the three component signal.
    [[nodiscard]]
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
        processWaveforms3C(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &z, 
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &n, 
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &e, 
        double samplingPeriod = 0.01);
/* 
    [[nodiscard]]
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
        processWaveforms3C(const std::vector<double> &z, 
                           const std::vector<double> &n, 
                           const std::vector<double> &e,
                           double samplingPeriod = 0.01);
*/
    /// Preprocess the input waveform.
    [[nodiscard]]
    std::vector<double> processWaveform(
         pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> &x,
         double samplingPeriod = 0.01);
/*
    [[nodiscard]]
    std::vector<double> processWaveform(const std::vector<double> &x, 
                                        const double samplingPeriod = 0.01);
*/
    /// @result The target sampling period in seconds.
    [[nodiscard]] double getTargetSamplingPeriod() const noexcept;

    ProcessData(const ProcessData &process) = delete;
    ProcessData(ProcessData &&process) noexcept = delete;
    ProcessData& operator=(const ProcessData &process) noexcept = delete;
    ProcessData& operator=(ProcessData &&process) = delete;
private:
    std::unique_ptr<UUSS::ThreeComponentPicker::ZRUNet::ProcessData> pImpl;
};
void initializeProcessing(pybind11::module &m);
}

}
#endif
