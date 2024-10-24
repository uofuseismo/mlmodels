#ifndef UUSS_DETECTORS_UNET_ONE_COMPONENT_P_INFERENCE_HPP
#define UUSS_DETECTORS_UNET_ONE_COMPONENT_P_INFERENCE_HPP
#include <vector>
#include <memory>
namespace UUSSMLModels::Detectors::UNetOneComponentP
{
/// @class Inference "inference.hpp" "models/detectors/uNetThreeComponentP/inference.hpp"
/// @brief Performs model inference for the vertical-component P UNet detector.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Inference
{
public:
    /// @brief Defines the model formats and frameworks supported for inference.
    enum ModelFormat
    {
        ONNX = 0 /*!< ONNX model format for use with OpenVINO. */
    };
    /// @brief Defines the device the inference should be performed on.
    enum Device
    {
        CPU = 0, /*!< Perform inference on the CPU. */
        GPU = 1  /*!< Perform inference on the GPU.
                      If this device is not available then the CPU will be used. */
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
 
    /// @brief The sampling rate of the input signal to inference and output
    ///        probability signal in Hz.
    [[nodiscard]] static double getSamplingRate() noexcept;
    /// @result The minimum signal length on which to perform inference.
    /// @note This is the number of samples in a single example.
    [[nodiscard]] static int getMinimumSignalLength() noexcept;
    /// @result The expected number of samples in a waveform.  This is
    ///         consistent with the length of the training examples.
    [[nodiscard]] static int getExpectedSignalLength() noexcept;
    /// @brief Checks if the seismogram is a valid length.  Nominally, this
    ///        means that the seismogram length is the requisite minimum
    ///        length (\c getMinimumSeismogramLength()) and is divisible by 16.
    /// @param[in] nSamples   The number of samples.
    /// @result True indicates that the seismogram length is valid.
    [[nodiscard]] static bool isValidSignalLength(int nSamples) noexcept;
    /// @result When training, the examples are only shifted +/- some lag around
    ///         about the center of the window.  Consequently, for a signal of
    ///         length \c getExpectedsignalLength() result.first is the first
    ///         sample where the probability signals should be believed and
    ///         result.second is the last sample where the signals should be
    ///         believed.
    [[nodiscard]] static std::pair<int, int> getCentralWindowStartEndIndex() noexcept;
    /// @}

    /// @name Model Evaluation
    /// @{

    /// @brief Predicts the probability of a sample corresponding to a
    ///        P arrival (1) or being noise (0).
    /// @param[in] vertical  The preprocessed signal on the vertical channel.
    /// @result The probability of the a sample corresponding to a P wave.
    /// @throws std::invalid_argument if the signal length is not equal
    ///         to \c getExpectedSignalLength() or any signals have an
    ///         inconsistent size.
    /// @throws std::runtime_error if \c isInitialized() is false.
    template<typename U>
    std::vector<U> predictProbability(const std::vector<U> &vertical) const;
    /// @brief Predicts the probability of a sample corresponding to a 
    ///        P arrival.  This uses a 500-sample sliding window that overlaps
    ///        with subsequent windows.
    /// @param[in] vertical  The preprocessed signal on the vertical channel.
    /// @note The first window will not zero-probabilities before the sample
    ///       254 (= 1008/2 - 500/2).  Likewise, the last window will not zero
    ///       the last 254 samples.
    /// @throws std::invalid_argument if the signal lengths are not consistent
    ///         or the signal length is not at least \c getMinimumSignalSize().
    /// @throws std::runtime_error if \c isInitialized() is false.
    template<typename U>
    std::vector<U> predictProbabilitySlidingWindow(const std::vector<U> &vertical) const;
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
