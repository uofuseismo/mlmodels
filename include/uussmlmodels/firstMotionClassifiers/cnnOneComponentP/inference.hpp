#ifndef UUSS_MLMODELS_FIRST_MOTION_CLASSIFIERS_CNN_ONE_COMPONENT_P_INFERENCE_HPP
#define UUSS_MLMODELS_FIRST_MOTION_CLASSIFIERS_CNN_ONE_COMPONENT_P_INFERENCE_HPP
#include <memory>
#include <vector>
#include <tuple>
namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP
{
/// @class Inference "inference.hpp" "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
/// @brief Predicts a first motion as being up, down, or unknown.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Inference
{
public:
    /// @brief Defines the model formats and frameworks supported for inference.
    enum ModelFormat
    {   
        ONNX = 0, /*!< ONNX model format for use with OpenVINO. */
        HDF5 = 1  /*!< The model weights are given in an HDF5 file. */
    };
    /// @brief Defines the device the inference should be performed on.
    enum Device
    {   
        CPU = 0, /*!< Perform inference on the CPU. */
        GPU = 1  /*!< Perform inference on the GPU.
                      If this device is not available then the CPU will be used. */
    };  
    /// @brief Defines the first motion of the arrival.
    /// @note The reference frame is positive up.
    enum FirstMotion : int
    {
        Unknown = 0, /*!< The arrival's first motion is indeterminate. */
        Up = +1,     /*!< The arrival's first motion is up. */
        Down =-1     /*!< The arrival's first motion is down. */
    };
public:
    /// @brief Constructors
    /// @{

    /// @brief Constructor.
    Inference();
    /// @brief Constructor where inference will be performed on a
    ///        desired device.
    explicit Inference(const Device device);
    /// @}

    /// @name Initialization
    /// @{

    /// @brief Loads the file containing the model.
    /// @param[in] fileName  The file name with the model weights.
    /// @param[in] format    The model format.
    void load(const std::string &fileName,
              const ModelFormat format = ModelFormat::ONNX);
    /// @result True indicates the model is initialized and ready for use.
    [[nodiscard]] bool isInitialized() const noexcept;
    /// @}

    /// @name Model Parameters
    /// @{
 
    /// @brief The sampling rate of the input signal to inference in Hz.
    [[nodiscard]] static double getSamplingRate() noexcept;
    /// @result The expected number of samples in a waveform.  This is
    ///         consistent with the length of the training examples.
    [[nodiscard]] static int getExpectedSignalLength() noexcept;

    /// @brief Sets the minimum threshold for a first-motion to be declared as
    ///        up or down.  This can be used to make the classifier more
    ///        conservative and require the posterior probability of an up or
    ///        down pick to be, say, 0.9.  The default is 1/3 which means
    ///        the most likely class is selected.
    /// @param[in] threshold  The posterior probability that a first-motion
    ///                       pick must exceed to be classified as an up or
    ///                       down pick.  For example,
    ///                       if this is 0.4 and the up probability is
    ///                       0.42, the down probability is 0.41, and the
    ///                       unknown probability is 0.17 then this will 
    ///                       classify as up since the up class exceeds the
    ///                       threshold and has the largest posterior
    ///                       probability.  However, if this is 0.6 and
    ///                       the up probability is 0.5, the down probability,
    ///                       is 0.2, and the unknown probability is 0.4 then
    ///                       this will classify to unknown since the up 
    ///                       probability did not exceed 0.6.  By default
    ///                       we approximate the Bayes's classifier and
    ///                       classify to the class with the largest
    ///                       posterior probability.
    /// @throw std::invalid_argument if the threshold is not in the range of
    ///        [0, 1].
    /// @note Setting this to a value less than 1/3 is effectively setting this
    ///       to a value of 1/3.  For example, if it is 0.2 and the posteriors
    ///       are (Up, Down, Unknown) = (0.32, 0.31, 0.33) then the 0.33
    ///       (unknown) class will be selected.
    void setProbabilityThreshold(double threshold);
    /// @result The posterior probability that a polarity must exceed to be
    ///         classified as up or down.
    [[nodiscard]] double getProbabilityThreshold() const noexcept;
    /// @}

    /// @name Model Evaluation
    /// @{

    /// @brief Predicts the probability of the first motion being
    ///        up, down, and unknown (respectively).
    /// @param[in] vertical  The preprocessed signal on the vertical channel.
    /// @result The probability of the first motion being up, down, and unkonwn
    ///         (respectively).
    /// @throws std::invalid_argument if the signal length is not equal
    ///         to \c getExpectedSignalLength().
    /// @throws std::runtime_error if \c isInitialized() is false.
    template<typename U>
    [[nodiscard]] std::tuple<U, U, U> predictProbability(const std::vector<U> &vertical) const;

    /// @brief Predicts the first motion being up/down/unknown.
    /// @param[in] vertical   The preprocessed signal on the vertical channel.
    /// @result The first motion.
    /// @throws std::invalid_argument if the signal length is not equal
    ///         to \c getExpectedSignalLength() or the threshold is not
    ///         in the range [0,1].
    /// @throws std::runtime_error if \c isInitialized() is false.
    template<typename U>
    [[nodiscard]] FirstMotion predict(const std::vector<U> &vertical) const;
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Releases the memory and resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~Inference();
    /// @}
private:
    class InferenceImpl;
    std::unique_ptr<InferenceImpl> pImpl;
};
}
#endif
