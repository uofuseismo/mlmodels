#ifndef UUSS_DETECTORS_UNET_THREE_COMPONENT_P_INFERENCE_HPP
#define UUSS_DETECTORS_UNET_THREE_COMPONENT_P_INFERENCE_HPP
#include <vector>
#include <memory>
namespace UUSSMLModels::Detectors::UNetThreeComponentP
{
/// @class Inference "inference.hpp" "models/detectors/uNetThreeComponentP/inference.hpp"
/// @brief Performs model inference.
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
        GPU = 1, /*!< Perform inference on the GPU.  If this device is not
                      available then the CPU will be used. */
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
    /// @brief The minimum signal length on which to perform inference.
    [[nodiscard]] static int getMinimumSignalLength() noexcept;
    /// @brief Checks if the seismogram is a valid length.  Nominally, this
    ///        means that the seismogram length is the requisite minimum
    ///        length (\c getMinimumSeismogramLength()) and is divisible by 16.
    /// @param[in] nSamples   The number of samples.
    /// @result True indicates that the seismogram length is valid.
    [[nodiscard]] bool isValidSignalLength(int nSamples) const noexcept;
    /// @}

    /// @name Model Evaluation
    /// @{

    /// @brief Predicts the probability of a sample corresponding to a
    ///        P arrival or being noise.
    /// @throws std::invalid_argument if \c isValidSignalLength() is false.
    /// @throws std::runtime_error if \c isInitialized() is false.
    std::vector<float> predictProbability(const std::vector<float> &vertical,
                                          const std::vector<float> &north,
                                          const std::vector<float> &east) const;
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
