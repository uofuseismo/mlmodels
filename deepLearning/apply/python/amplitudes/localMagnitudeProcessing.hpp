#ifndef PUUSSMLMODEL_AMPLITUDES_LOCALMAGNITUDEPROCESSING_HPP
#define PUUSSMLMODEL_AMPLITUDES_LOCALMAGNITUDEPROCESSING_HPP
#include <memory>
#include <pybind11/pybind11.h>
// Forward declarations
namespace UUSS::Amplitudes
{
    class LocalMagnitudeProcessing;
}
namespace PUUSSMLModels::Amplitudes
{
class LocalMagnitudeProcessing
{
public:
    /// C'tor
    LocalMagnitudeProcessing();
    /// Destructor
    ~LocalMagnitudeProcessing();
    /// Preprocess the input waveform.
    [[nodiscard]]
    std::vector<double> processWaveform(const bool isVelocity, //std::string &channel,
                                        const double gain,
                                        const std::vector<double> &x, 
                                        const double samplingPeriod = 0.01);
    [[nodiscard]]
    std::vector<int> computeMinMaxSignal(const std::vector<double> &x);
    /// @result The target sampling period in seconds.
    [[nodiscard]] double getTargetSamplingPeriod() const noexcept;

    LocalMagnitudeProcessing(const LocalMagnitudeProcessing &process) = delete;
    LocalMagnitudeProcessing(LocalMagnitudeProcessing &&process) noexcept = delete;
    LocalMagnitudeProcessing& operator=(const LocalMagnitudeProcessing &process) noexcept = delete;
    LocalMagnitudeProcessing& operator=(LocalMagnitudeProcessing &&process) = delete;
private:
    std::unique_ptr<UUSS::Amplitudes::LocalMagnitudeProcessing> pImpl;
};
void initializeLocalMagnitudeProcessing(pybind11::module &m);
}
#endif
