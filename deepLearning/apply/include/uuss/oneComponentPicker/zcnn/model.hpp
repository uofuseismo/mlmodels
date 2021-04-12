#ifndef UUSSMLMODELS_ONECOMPONENTPICKER_ZCNN_MODEL_HPP
#define UUSSMLMODELS_ONECOMPONENTPICKER_ZCNN_MODEL_HPP
#include <memory>
#include "uuss/enums.hpp"
namespace UUSS::OneComponentPicker::ZCNN
{
/// @brief This class is for applying the convolutional neural network
///        picker to the vertical channel.  This was architecture
///        was defined P Wave Arrival Picking and First-Motion Polarity
///        Determination With Deep Learning (Ross et al., 2016).
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
template<UUSS::Device E = UUSS::Device::CPU>
class Model
{
public:
    /// @name Constructors
    /// @{
    /// @brief Default constructor.
    Model();
    /// @brief Move constructor.
    /// @param[in,out] model  The ML model from which to initialize this moden
    ///                       on exit model's behavior is undefined.
    Model(Model &&model) noexcept;
    /// @}

    /// @name Operators
    /// @{
    /// @brief Move assignment operator.
    /// @param[in,out] model  The ML model whose memory will be moved to this.
    ///                       On exit, model's behavior is undefined.
    /// @result The memory from model moved to this.
    Model& operator=(Model &&model) noexcept;
    /// @}

    /// @name Destructors
    /// @{
    /// @brief Destructor.
    ~Model();
    /// @}

    /// @brief Loads the network weights from an HDF5 file.
    /// @param[in] fileName  The name of the HDF5 file with the model
    ///                      coefficients.
    /// @param[in] verbose   If true then extra information will be printed
    ///                      while loading the weights.  This is useful for
    ///                      debugging.
    /// @throws std::invalid_argument if the HDF5 file does not exist or is
    ///         improperly formatted.
    void loadWeightsFromHDF5(const std::string &fileName,
                             bool verbose = false);
    /// @result True indicates that the model coefficients are set.
    bool haveModelCoefficients() const noexcept;
    /// @result The expected input signal length.
    /// @note The first underlying fully-connected layer requires a flattened
    ///       layer of expected length.  Therefore, the input signal must have
    ///       a specified length as to conform with the model architecture.
    int getSignalLength() const noexcept;
    /// @result The assumed sampling p;eriod of the input signal in seconds.
    double getSamplingPeriod() const noexcept;

    /// @brief Computes the probability of a waveform's polarity as being
    ///        up, down, or unknown.
    /// @param[in] nSignals  The number of signals.  This must be positive.
    /// @param[in] nSamplesInSignal  The number of samples in each signal.
    ///                              This must match \c getSignalLength(). 
    /// @param[in] z   The vertical channel signals from which to compute
    ///                the respective proababilities.  This is a row major
    ///                matrix with dimension [nSignals x nSamplesInSignal].
    /// @param[out] probaUp  The probability of an up polarity for the is'th
    ///                      signal.  This is an array whose dimension is
    ///                      [nSignals].
    /// @param[out] probaDown  The probability of a down polarity for the is'th
    ///                        signal.  This is an array whose dimension is
    ///                        [nSignals].
    /// @param[out] probaUnknown  The probability of unknown polarity for
    ///                           the is'th signal.  This is an array whose
    ///                           dimension is [nSignals].
    /// @param[in] batchSize  The number of signals for torch to process
    ///                      simultaneously.  This must be positive.
    /// @throws std::invalid_argument if nSignals is not positive,
    ///         nSamplesInSignal is not valid, any array is NULL.
    /// @throws std::runtime_error if the coefficients are not set.
    void predict(int nSignals, int nSamplesInSignal,
                 const float z[],
                 float *pickTimes[],
                 int batchSize = 32) const;
    /// @copydoc predict
    void predict(int nSignal, int nSamplesInSignal,
                 const double z[],
                 double *pickTimes[],
                 int batchSize = 32) const;
    /// @brief Computes the pick time in seconds from the trace's start.
    /// @param[in] nSamples   The number of samples in the signal.  
    ///                       This must match \c getSignalLength().
    /// @param[in] z  The vertical channel signal on which to make a P
    ///               pick.
    /// @result The pick time in seconds relative from the trace start. 
    /// @throws std::invalid_argument if nSamples does not equal
    ///         \c getSignalLength() or z is NULL.
    float predict(int nSamples, const float z[]) const;
    /// @copydoc predict
    double predict(int nSamples, const double z[]) const;

    Model(const Model &model) = delete;
    Model& operator=(const Model &model) = delete;
private:
    class ZCNNImpl;
    std::unique_ptr<ZCNNImpl> pImpl;
};
}
#endif
