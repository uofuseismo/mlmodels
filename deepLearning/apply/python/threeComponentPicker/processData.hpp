#ifndef PUUSSMLMODEL_THREECOMPONENTPICKER_PROCESSDATA_HPP
#define PUUSSMLMODEL_THREECOMPONENTPICKER_PROCESSDATA_HPP
#include <memory>
#include <pybind11/pybind11.h>
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
        processWaveform3C(const std::tuple<std::vector<double>,
                                           std::vector<double>,
                                           std::vector<double>> &waveforms,
                          const double samplingPeriod = 0.01);
    /// Preprocess the input waveform.
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
        processWaveform3C(const std::tuple<std::vector<double>,
                                           std::vector<double>,
                                           std::vector<double>> &waveforms,
                          const double samplingPeriod = 0.01);
    /// Preprocess the input waveform.
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
    std::unique_ptr<UUSS::ThreeComponentPicker::ZRUNet::ProcessData> pImpl;
};
void initializeProcessing(pybind11::module &m);
}

}
#endif
