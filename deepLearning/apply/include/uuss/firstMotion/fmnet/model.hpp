#ifndef UUSSMLMODELS_FIRSTMOTION_FMNET_MODEL_HPP
#define UUSSMLMODELS_FIRSTMOTION_FMNET_MODEL_HPP
#include <memory>
#include "uuss/enums.hpp"
namespace UUSS::FirstMotion::FMNet
{
/// @class Model "model.hpp" "uuss/firstMotion/fmnet/model.hpp"
/// @brief This class is for applying the first motion model to a pre-processed
///        vertical channel.
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
    /// @param[in,out] model  The model from which to initialize this class.
    ///                       On exit, model's behavior is undefined.
    Model(Model &&model) noexcept;
    /// @}

    /// @name Operators
    /// @{
    /// @brief Move assignment operator.
    /// @param[in,out] model  The model whose memory will be moved to this.
    ///                       On exit, model's behavior is undefined.
    Model& operator=(Model &&model) noexcept;
    /// @}

    /// @name Destructors
    /// @{
    /// @brief Destructor.
    ~Model();
    /// @}

    /// @name Initialization
    /// @{
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
    /// @brief Loads the model weights from a typical torchscript output.
    /// @param[in] fileName  The name of the pt file containing the fitted
    ///                      model coefficients.
    void loadWeightsFromPT(const std::string &fileName);
    /// @result True indicates that the model coefficients are set.
    [[nodiscard]] bool haveModelCoefficients() const noexcept;
    /// @brief Gets the expected input signal length.
    /// @result The expected input signal length.
    /// @note The first underlying fully-connected layer requires a flattened
    ///       layer of expected length.  Therefore, the input signal must have
    ///       a specified length as to conform with the model architecture.
    [[nodiscard]] int getSignalLength() const noexcept;
    /// @brief Sets the minimum threshold for a polarity to be declared as up
    ///        or down.  This can be used to make the classifier more
    ///        conservative and require the posterior probability of an or
    ///        down pick to be, say, 0.9.  The default is 1/3 which means
    ///        the most likely class is selected.
    /// @param[in] threshold   The posterior probability that a polarity pick
    ///                        must exceed to be classified as an up or
    ///                        down pick.
    /// @throw std::invalid_argument if the threshold is not in the range of
    ///        [0, 1].
    /// @note Setting this to a value less than 1/3 is effectively setting this
    ///       to a value of 1/3.  For example, if it is 0.2 and the posteriors
    ///       are (Up, Down, Unknown) = (0.32, 0.31, 0.33) then the 0.33
    ///       (unkonwn) class will be selected.
    void setPolarityThreshold(double threshold);
    /// @result The posterior probability that a polarity must exceed to be
    ///         classified as up or down.
    [[nodiscard]] double getPolarityThreshold() const noexcept;
    /// @}

    /// @name Inference
    /// @{
    /// @brief Predicts whether a probablity is up, down, or unknown.
    /// @param[in] nSamples  The number of samples in z.  This must match
    ///                      \c getSignalLength().
    /// @param[in] z         The vertical channel on which to make a polarity
    ///                      classification.  The pick should be near the
    ///                      center of the window.  This is an array whose
    ///                      dimension is [nSamples].
    /// @result +1 indicates a positive polarity, -1 indicates a negative
    ///         polarity, and 0 indicates an unknown polarity.
    /// @throws std::invalid_argument if nSamples does not equal
    ///         \c getSignalLength() or z is NULL.
    /// @sa \c setPolarityThreshold()
    [[nodiscard]] int predict(int nSamples, const float z[]) const;

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
    ///                       simultaneously.  This must be positive.
    /// @throws std::invalid_argument if nSignals is not positive,
    ///         nSamplesInSignal is not valid, any array is NULL.
    /// @throws std::runtime_error if the coefficients are not set.
    void predictProbability(int nSignals, int nSamplesInSignal,
                            const float z[],
                            float *probaUp[],
                            float *probaDown[],
                            float *probaUnknown[],
                            int batchSize = 32) const;
    /// @brief Computes the probability of a waveform's polarity as being up,
    ///        down, and unknown.
    /// @param[in] nSamples   The number of samples in z.  This must match
    ///                       \c getSignalLength().
    /// @param[in] z          The vertical channel on which to make a polarity
    ///                       classification.  The pick should be near the
    ///                       center of the window.  This is an array whose
    ///                       dimension is [nSamples].
    /// @param[out] pUp       The probability of the polarity being up.
    /// @param[out] pDown     The probability of the polarity being down.
    /// @param[out] pUnknown  The probability of the polarity being uknonwn.
    /// @note pUp + pDown + pUnkonwn = 1.
    /// @throws std::invalid_argument if nSamples does not equal
    ///         \c getSignalLength() or z is NULL.
    void predictProbability(int nSamples, const float z[],
                            float *pUp, float *pDown, float *pUnknown) const;
    /*! @copydoc predictProbability */
    //void predictProbability(int nSamples, const double z[],
    //                        double *pUp, double *pDown, double *pUnknown) const;
    /// @}

    /// Remove some functionality.
    Model(const Model &model) = delete;
    Model& operator=(const Model &model) = delete;
private:
    class FMNetImpl;
    std::unique_ptr<FMNetImpl> pImpl;
};
}
#endif
