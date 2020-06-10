#ifndef UUSSMLMODELS_FIRSTMOTION_FMNET_MODEL_HPP
#define UUSSMLMODELS_FIRSTMOTION_FMNET_MODEL_HPP
#include <memory>
#include "uuss/enums.hpp"
namespace UUSS::FirstMotion::FMNet
{
/*!
 * @brief This class is for applying the first motion model to a pre-processed
 *        vertical channel.
 * @copyright Ben Baker (University of Utah) distributed under the MIT license.
 */
template<UUSS::Device E = UUSS::Device::CPU>
class Model
{
public:
    /*! @name Constructors
     * @{
     */
    /*!
     * @brief Default constructor.
     */
    Model();
    /*! @} */

    /*! @name Destructors
     * @{
     */
    /*!
     * @brief Destructor.
     */
    ~Model();
    /*! @} */

    /*!
     * @brief Sets the minimum threshold for a polarity to be declared as up
     *        or down.  This can be used to make the classifier more
     *        conservative and require the posterior probability of an or
     *        down pick to be, say, 0.9.  The default is 1/3 which means
     *        the most likely class is selected.
     * @param[in] threshold   The posterior probability that a polarity pick
     *                        must exceed to be classified as an up or
     *                        down pick.
     * @throw std::invalid_argument if the threshold is not in the range of
     *        [0, 1].
     * @note Setting this to a value less than 1/3 is effectively setting this
     *       to a value of 1/3.  For example, if it is 0.2 and the posteriors
     *       are (Up, Down, Unknown) = (0.32, 0.31, 0.33) then the 0.33
     *       (unkonwn) class will be selected.
     */
    void setPolarityThreshold(double threshold);
    /*!
     * @brief Gets the polarity threshold.
     * @result The posterior probability that a polarity must exceed to be
     *         classified as up or down.
     */
    double getPolarityThreshold() const noexcept;
    /*!
     * @brief Loads the network weights from an HDF5 file.
     * @param[in] fileName  The name of the HDF5 file with the model
     *                      coefficients.
     * @param[in] verbose   If true then extra information will be printed
     *                      while loading the weights.  This is useful for
     *                      debugging.
     * @throws std::invalid_argument if the HDF5 file does not exist or is
     *         improperly formatted.
     */
    void loadWeightsFromHDF5(const std::string &fileName,
                             bool verbose = false);
    /*!
     * @brief Indicates whether or not the model coefficients are set.
     * @brief True indicates that the model coefficients are set.
     */
    bool haveModelCoefficients() const noexcept;
    /*!
     * @brief Gets the expected input signal length.
     * @result The expected input signal length.
     * @note The first underlying fully-connected layer requires a flattened
     *       layer of expected length.  Therefore, the input signal must have
     *       a specified length as to conform with the model architecture.
     */
    int getSignalLength() const noexcept;

    /*!
     * @brief Predicts whether a probablity is up, down, or unknown.
     * @param[in] nSamples   The number of samples in z.  This must match
     *                       \c getSignalLength().
     * @param[in] z          The vertical channel on which to make a polarity
     *                       classification.  The pick should be near the
     *                       center of the window.  This is an array whose
     *                       dimension is [nSamples].
     * @result +1 indicates a positive polarity, -1 indicates a negative
     *         polarity, and 0 indicates an unknown polarity.
     * @throws std::invalid_argument if nSamples does not equal
     *         \c getSignalLength() or z is NULL.
     * @sa \c setPolarityThreshold()
     */
    int predict(int nSamples, const float z[]) const;
    /*!
     * @brief Computes the probability of a waveform's polarity as being up,
     *        down, and unknown.
     * @param[in] nSamples   The number of samples in z.  This must match
     *                       \c getSignalLength().
     * @param[in] z          The vertical channel on which to make a polarity
     *                       classification.  The pick should be near the
     *                       center of the window.  This is an array whose
     *                       dimension is [nSamples].
     * @param[out] pUp       The probability of the polarity being up.
     * @param[out] pDown     The probability of the polarity being down.
     * @param[out] pUnknown  The probability of the polarity being uknonwn.
     * @note pUp + pDown + pUnkonwn = 1.
     * @throws std::invalid_argument if nSamples does not equal
     *         \c getSignalLength() or z is NULL.
     */
    void predictProbability(int nSamples, const float z[],
                            float *pUp, float *pDown, float *pUnknown) const;
    /*! @copydoc predictProbability */
    //void predictProbability(int nSamples, const double z[],
    //                        double *pUp, double *pDown, double *pUnknown) const;
private:
    class FMNetImpl;
    std::unique_ptr<FMNetImpl> pImpl;
};
}
#endif
