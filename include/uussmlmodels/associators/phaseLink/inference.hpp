#ifndef UUSS_ASSOCIATORS_PHASE_LINK_INFERENCE_HPP
#define UUSS_ASSOCIATORS_PHASE_LINK_INFERENCE_HPP
#include <vector>
#include <memory>
namespace UUSSMLModels::Associators::PhaseLink
{
 class Pick;
 class Arrival;
}
namespace UUSSMLModels::Associators::PhaseLink
{
enum class Region
{
    Utah,
    Yellowstone
};
/// @class Inference "inference.hpp" "mlmodels/associators/phaseLink/inference.hpp"
/// @brief Performs model inference for the PhaseLink model.
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

    Inference() = delete;
    /// @brief Constructor.
    explicit Inference(const Region region);
    /// @brief Constructor where inference will be performed on a
    ///        desired device.
    Inference(const Region region, const Device device);
    /// @}

    /// @name Initialization
    /// @{

    /// @brief Loads the file containing the model.
    /// @param[in] fileName  The file name with the model weights.
    /// @param[in] format    The model format.
    void load(const std::string &fileName,
              ModelFormat format = ModelFormat::ONNX);
    /// @result True indicates the model is initialized and ready for use.
    [[nodiscard]] bool isInitialized() const noexcept;

    /// @result Gets the number of features.
    [[nodiscard]] static int getNumberOfFeatures() noexcept;
    /// @result Gets the simulation size.
    [[nodiscard]] int getSimulationSize() const noexcept;
    /// @}

    void associate(const std::vector<Pick> &picks, const double threshold = 0.5) const;

    /// @brief Predicts the posterior probability of each phase being linked
    ///        to the root arrival.
    /// @param[in] X   The feature matrix in row major format.
    /// @result The posterior probability of each phase being linked to the
    ///         root pick.  This has dimension \c getSimulationSize().
    /// @throws std::invalid_argument if X.size() != getSimulationSize() x 5.
    template<typename U>
    std::vector<U> predictProbability(const std::vector<U> &X) const;
    /// @brief Predicts the posterior probability of each phase being linked
    ///        to the root arrival.
    /// @param[in] nPicks  The number of picks.
    /// @param[in] X       The feature matrix in row major format.
    /// @result The posterior probability of each phase being linked to the
    ///         root pick.  This has dimension \c nPicks.
    /// @throws std::invalid_argument if X.size() < getNumberOfFeatures() or 
    ///         the input exceeds getSimulationSize() x getNumberOfFeatures().
    template<typename U>
    std::vector<U> predictProbability(const int nPicks, const std::vector<U> &X) const;
 
    /// @brief Predicts whether each phase is linked to the root arrival.
    /// @param[in] nPicks  The number of picks.
    /// @param[in] X       The feature matrix in row major format with dimension
    ///                    [nPicks x getFeatureSize()].
    /// @param[in] threshold   If the i'th phase's posterior probability
    ///                        exceeds this value then it will be mapped to
    ///                        a positive link (1) otherwise it is not
    ///                        linked to the root arrival.
    /// @result 1 indicates this pick is linked to the root face.  This has
    ///         dimension [nPicks].
    /// @throws std::invalid_argument if X.size() < getNumberOfFeatures() or 
    ///         the input exceeds getSimulationSize() x getNumberOfFeatures().
    template<typename U>
    std::vector<int> predict(const int nPicks, const std::vector<U> &X, double threshold = 0.5) const;
    /// @brief Predicts whether each phase is linked to the root arrival.
    /// @param[in] nPicks  The number of picks.
    /// @param[in] X       The feature matrix in row major format with dimension
    ///                    [getSimulationSize() x getFeatureSize()].
    /// @param[in] threshold   If the i'th phase's posterior probability
    ///                        exceeds this value then it will be mapped to
    ///                        a positive link (1) otherwise it is not
    ///                        linked to the root arrival.
    /// @result 1 indicates this pick is linked to the root face.  This has
    ///         dimension [getSimulationSize()].
    /// @throws std::invalid_argument if X.size() < getNumberOfFeatures() or 
    ///         the input exceeds getSimulationSize() x getNumberOfFeatures().
    template<typename U>
    std::vector<int> predict(const std::vector<U> &X, double threshold = 0.5) const;

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
