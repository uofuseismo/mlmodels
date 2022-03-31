#ifndef UUSSMLMODELS_THREECOMPONENTPICKER_ZCNN_MODEL_HPP
#define UUSSMLMODELS_THREECOMPONENTPICKER_ZCNN_MODEL_HPP
#include <memory>
#include "uuss/enums.hpp"
namespace UUSS::ThreeComponentPicker::ZCNN
{
/// @brief This class is for applying the convolutional neural network
///        picker to a three-component signal.  This is an extension of
///        the architecture defined by P Wave Arrival Picking and
///        First-Motion Polarity Determination With Deep Learning
///        (Ross et al., 2016).  It is mainly used for picking S waves
///        at UUSS.
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
    [[nodiscard]] bool haveModelCoefficients() const noexcept;
    /// @result The expected input signal length.
    /// @note The first underlying fully-connected layer requires a flattened
    ///       layer of expected length.  Therefore, the input signal must have
    ///       a specified length as to conform with the model architecture.
    [[nodiscard]] static int getSignalLength() noexcept;
    /// @result The assumed sampling p;eriod of the input signal in seconds.
    [[nodiscard]] static double getSamplingPeriod() noexcept;

    /// @brief Computes the pick correction to add to the original pick.
    /// @param[in] nSignals  The number of signals.  This must be positive.
    /// @param[in] nSamplesInSignal  The number of samples in each signal.
    ///                              This must match \c getSignalLength(). 
    /// @param[in] vertical   The vertical channel signals from which to compute
    ///                       the respective proababilities.  This is a row
    ///                       major matrix with dimension
    ///                       [nSignals x nSamplesInSignal].
    /// @param[out] pickTimes The pick corrections to add to the pick.
    /// @param[in] batchSize   The number of signals for torch to process
    ///                        simultaneously.  This must be positive.
    /// @throws std::invalid_argument if nSignals is not positive,
    ///         nSamplesInSignal is not valid, any array is NULL.
    /// @throws std::runtime_error if the coefficients are not set.
    void predict(int nSignals, int nSamplesInSignal,
                 const float vertical[],
                 const float north[],
                 const float east[],
                 float *pickTimes[],
                 int batchSize = 32) const;
    /// @copydoc predict
    void predict(int nSignal, int nSamplesInSignal,
                 const double vertical[],
                 const double north[],
                 const double east[],
                 double *pickTimes[],
                 int batchSize = 32) const;
    /// @brief Computes the pick correction in seconds to add to the
    ///        original pick.
    /// @param[in] nSamples   The number of samples in the signal.  
    ///                       This must match \c getSignalLength().
    /// @param[in] vertical  The signal on the vertical channel.
    /// @param[in] north     The signal on the north channel.
    /// @param[in] east      The signal on the east channel.
    /// @result The pick time in seconds relative from the trace start. 
    /// @throws std::invalid_argument if nSamples does not equal
    ///         \c getSignalLength() or z is NULL.
    [[nodiscard]] float predict(int nSamples,
                                const float vertical[],
                                const float north[],
                                const float east[]) const;
    /// @copydoc predict
    [[nodiscard]] double predict(int nSamples,
                                 const double vertical[],
                                 const double north[],
                                 const double east[]) const;

    Model(const Model &model) = delete;
    Model& operator=(const Model &model) = delete;
private:
    class ZCNN3CImpl;
    std::unique_ptr<ZCNN3CImpl> pImpl;
};
}
#endif
