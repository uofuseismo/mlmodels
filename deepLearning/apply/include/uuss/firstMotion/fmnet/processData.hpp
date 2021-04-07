#ifndef UUSSMLMODELS_FIRSTMOTION_FMNET_PROCESSDATA_HPP
#define UUSSMLMODELS_FIRSTMOTION_FMNET_PROCESSDATA_HPP
#include <memory>
namespace UUSS::FirstMotion::FMNet
{
/// @class ProcessData "processData.hpp" "uuss/firstMotion/fmnet/processData.hpp"
/// @brief This class is for applying a uniform data processing strategy to
///        the first motion data.  You should call this prior to evaluating
///        the model.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class ProcessData
{
public:
    /// @brief Constructor.
    ProcessData();
    /// @brief Move constructor.
    /// @param[in,out] process  The process data class from which to initialize
    ///                         this class.  On exit, process's behavior is
    ///                         undefined.
    ProcessData(ProcessData &&process) noexcept;
    /// @brief Move assignment.
    /// @param[in,out] process  The process data class from which to initialize
    ///                         this class.  On exit, process's behavior is
    ///                         undefined.
    /// @result The memory from process moved to this. 
    ProcessData& operator=(ProcessData &&process) noexcept; 
    /// @brief Destructor.
    ~ProcessData();
    /// @brief Processes the data.
    /// @param[in] npts   The number of samples in the time series.
    /// @param[in] samplingPeriod  The sampling period of the data in seconds.
    ///                           This must be positive.
    /// @param[in] data   The data to process.  This is an array whose dimension
    ///                   is [npts].
    /// @param[out] processedData  The vector containing the processed data.
    ///                            The sampling period can be ascertained from
    ///                            \c getTargetSamplingPeriod().
    /// @throws std::invalid_argument if npts is not positive, data is NULL,
    ///         or sampling period is too small.
    void processWaveform(int npts, double samplingPeriod,
                         const double data[],
                         std::vector<double> *processedData);
    /// @copydoc processWaveform
    void processWaveform(int npts, double samplingPeriod,
                         const float data[],
                         std::vector<float> *processedData);
    /// @result The sampling period of the processed signals.
    [[nodiscard]] double getTargetSamplingPeriod() const;
    ProcessData& operator=(const ProcessData &p) = delete;
private:
    class ProcessDataImpl;
    std::unique_ptr<ProcessDataImpl> pImpl;
};
}
#endif
