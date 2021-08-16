#ifndef UUSS_THREECOMPONENTPICKER_ZCNN_PROCESSDATA_HPP
#define UUSS_THREECOMPONENTPICKER_ZCNN_PROCESSDATA_HPP
#include <vector>
#include <memory>
namespace UUSS::ThreeComponentPicker::ZCNN
{
/// @class ProcessData "processData.hpp" "uuss/threeComponentPicker/zcnn/processData.hpp"
/// @brief This class is for applying a uniform data processing strategy to
///        the data.  This should be called prior to applying the model.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class ProcessData
{
public:
    /// @name Constructors
    /// @{
    /// @brief Constructor.
    ProcessData();
    /// @brief Move constructor.
    /// @param[in,out] process  The data processing class from which to
    ///                         initialize this class.  On exit, process's
    ///                         behavior is undefined.
    ProcessData(ProcessData &&process) noexcept;
    /// @}

    /// @name Operators
    /// @{
    /// @brief Moved assignment operator.
    /// @param[in,out] process  The data processing class whose memory will be
    ///                         moved to this.  On exit, process's behavior is
    ///                         undefined.
    /// @result The memory from process moved to this.
    ProcessData& operator=(ProcessData &&process) noexcept;
    /// @}

    /// @name Destructors
    /// @{
    /// @brief Destructor.
    ~ProcessData();
    /// @}

    /// @brief Processes the data.
    /// @param[in] waveforms   The waveforms stored as a (vertical, north, east)
    ///                        tuple.
    /// @result The processed waveforms stored as (vertical, north, east) tuple.
    /// @throws std::invalid_argument if the sampling period is invalid or
    ///         any signals are empty.
    /// @note If you permute the order of the input tuple then the output
    ///       tuple will be in that order, e.g., an (e,n,z) will result
    ///       in an (e,n,z) ordered output tuple.
    template<typename U>
    [[nodiscard]]
    std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
    processWaveforms(const std::tuple<const std::vector<U> &,
                                      const std::vector<U> &,
                                      const std::vector<U> &> &waveforms,
                     double samplingPeriod);
    /// @brief Processes the data.
    /// @param[in] npts   The number of samples in the time series.
    /// @param[in] samplingPeriod  The sampling period of the data in seconds.
    ///                            This must be positive.
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
    [[nodiscard]] double getTargetSamplingPeriod() const noexcept;

    ProcessData(const ProcessData &p) = delete;
    ProcessData& operator=(const ProcessData &p) = delete;
private:
    class ProcessDataImpl;
    std::unique_ptr<ProcessDataImpl> pImpl;
};
}
#endif
