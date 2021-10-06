#ifndef UUSSMLMODELS_AMPLITUDES_LOCALMAGNITUDEPROCESSING_HPP
#define UUSSMLMODELS_AMPLITUDES_LOCALMAGNITUDEPROCESSING_HPP
#include <memory>
#include <string>
#include <vector>
namespace UUSS::Amplitudes
{
/// @class ProcessData "processData.hpp" "uuss/firstMotion/fmnet/processData.hpp"
/// @brief This class is for applying the preprocessing done by Jiggle when
///        computing local magnitudes.  This effectively demeans the signal,
///        applies a 5 pct taper, for reasons I do not understand applies a
///        high-pass filter to the accelerometers but not the velocity sensors,
///        and then passes the resulting signal through a time-domain 
///        Wood-Anderson filter.  See Kanamori et al., 1998 for details.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class LocalMagnitudeProcessing
{
public:
    /// @brief Constructor.
    LocalMagnitudeProcessing();
    /// @brief Move constructor.
    /// @param[in,out] process  The process data class from which to initialize
    ///                         this class.  On exit, process's behavior is
    ///                         undefined.
    LocalMagnitudeProcessing(LocalMagnitudeProcessing &&process) noexcept;
    /// @brief Move assignment.
    /// @param[in,out] process  The process data class from which to initialize
    ///                         this class.  On exit, process's behavior is
    ///                         undefined.
    /// @result The memory from process moved to this. 
    LocalMagnitudeProcessing& operator=(LocalMagnitudeProcessing &&process) noexcept; 
    /// @brief Destructor.
    ~LocalMagnitudeProcessing();
    /// @brief Processes the data.
    /// @param[in] channel  The channel name.  What is going to happen is 
    ///                     `accelerometers' will be identified with the
    ///                     second letter `N' and velocimeters are will be
    ///                     identified with the second letter `H'.
    /// @param[in] gain     This is the simple response
    /// @param[in] npts     The number of samples in the time series.
    /// @param[in] samplingPeriod  The sampling period of the data in seconds.
    ///                            This must be positive.
    /// @param[in] data   The data to process.  This is an array whose dimension
    ///                   is [npts].
    /// @param[out] processedData  The vector containing the processed data.
    ///                            The sampling period can be ascertained from
    ///                            \c getTargetSamplingPeriod().
    /// @throws std::invalid_argument if npts is not positive, data is NULL,
    ///         or sampling period is too small.
    void processWaveform(const std::string &channel, double gain,
                         int npts, double samplingPeriod,
                         const double data[],
                         std::vector<double> *processedData);
    /// @copydoc processWaveform
    void processWaveform(const std::string &channel, double gain,
                         int npts, double samplingPeriod,
                         const float data[],
                         std::vector<float> *processedData);
    /// @result The sampling period of the processed signals.
    [[nodiscard]] double getTargetSamplingPeriod() const noexcept;
    LocalMagnitudeProcessing& operator=(const LocalMagnitudeProcessing &p) = delete;
private:
    class LocalMagnitudeProcessingImpl;
    std::unique_ptr<LocalMagnitudeProcessingImpl> pImpl;
};
}
#endif
