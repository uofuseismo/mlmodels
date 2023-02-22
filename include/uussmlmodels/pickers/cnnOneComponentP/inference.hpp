#ifndef UUSS_MLMODELS_PICKERS_CNN_ONE_COMPONENT_P_INFERENCE_HPP
#define UUSS_MLMODELS_PICKERS_CNN_ONE_COMPONENT_P_INFERENCE_HPP
#include <memory>
#include <vector>
#include <tuple>
namespace UUSSMLModels::Pickers::CNNOneComponentP
{
/// @class Inference "inference.hpp" "uussmlmodels/pickers/cnnOneComponentP/inference.hpp"
/// @brief Computes the correction to add to a pick.
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
    [[nodiscard]] U predict(const std::vector<U> &vertical) const;
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
