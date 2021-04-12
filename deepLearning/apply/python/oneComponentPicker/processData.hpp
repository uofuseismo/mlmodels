#ifndef PUUSSMLMODEL_ONECOMPONENTPICKER_PROCESSDATA_HPP
#define PUUSSMLMODEL_ONECOMPONENTPICKER_PROCESSDATA_HPP
#include <memory>
#include <pybind11/pybind11.h>
// Forward declarations
namespace UUSS::OneComponentPicker
{
    namespace ZCNN
    {
        class ProcessData;
    }
}
namespace PUUSSMLModels::OneComponentPicker
{
namespace ZCNN
{
class ProcessData
{
public:
    /// C'tor
    ProcessData();
    /// Destructor
    ~ProcessData();
    /// Preprocess the input waveform.
    [[nodiscard]]
    std::vector<double> processWaveform(const std::vector<double> &x,
                                        const double samplingRate = 100);
    /// @result The target sampling period in seconds.
    [[nodiscard]] double getTargetSamplingPeriod() const noexcept;

    ProcessData& operator=(const ProcessData &process) = delete;
private:
    std::unique_ptr<UUSS::OneComponentPicker::ZCNN::ProcessData> pImpl;
};
void initializeProcessing(pybind11::module &m);
} // End ZCNN
}
#endif
