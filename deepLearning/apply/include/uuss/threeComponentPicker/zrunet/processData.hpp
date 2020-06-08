#ifndef UUSS_THREECOMPONENTPICKER_ZRUNET_PROCESSDATA_HPP
#define UUSS_THREECOMPONENTPICKER_ZRUNET_PROCESSDATA_HPP
#include <vector>
#include <memory>
namespace UUSS::ThreeComponentPicker::ZRUNet
{
/*!
 * @brief This class is for applying a uniform data processing strategy to
 *        the data.  This should be called prior to applying the model.
 * @copyright Ben Baker (University of Utah) distributed under the MIT license.
 */
class ProcessData
{
public:
    /// Constructor
    ProcessData();
    /// Destructor
    ~ProcessData();
    /*!
     * @brief Processes the data.
     * @param[in] npts   The number of samples in the time series.
     * @param[in] samplingPeriod  The sampling period of the data in seconds.
     *                            This must be positive.
     * @param[in] data   The data to process.  This is an array whose dimension
     *                   is [npts].
     * @param[out] processedData  The vector containing the processed data.
     *                            The sampling period can be ascertained from
     *                            \c getTargetSamplingPeriod().
     * @throws std::invalid_argument if npts is not positive, data is NULL,
     *         or sampling period is too small.
     */
    void processWaveform(int npts, double samplingPeriod,
                         const double data[],
                         std::vector<double> *processedData);
    /*! @copydoc processWaveform */
    void processWaveform(int npts, double samplingPeriod,
                         const float data[],
                         std::vector<float> *processedData);
    /*!
     * @brief Gets the target sampling period.
     * @result The sampling period of the processed signals.
     */
    double getTargetSamplingPeriod() const;
private:
    ProcessData(const ProcessData &p) = delete;
    ProcessData& operator=(const ProcessData &p) = delete;
    class ProcessDataImpl;
    std::unique_ptr<ProcessDataImpl> pImpl;
};
}
#endif
