#ifndef UUSS_THREE_COMPONENT_PICKER_ZRUNET_MODEL_HPP
#define UUSS_THREE_COMPONENT_PICKER_ZRUNET_MODEL_HPP
#include <memory>
#include "uuss/enums.hpp"
// Forward declarations
namespace UUSS::ThreeComponentPicker::ZRUNet
{
/// @class Model "model.hpp" "uuss/threeComponentPicker/zrunet/model.hpp"
/// @brief Defines the three component picker that leverages the U-Net
///        architecture.
/// @note This architecture is slightly modified from Zach Ross's architecture.
/// @copyright Ben Baker distributed under the MIT license.
template<UUSS::Device E = UUSS::Device::CPU>
class Model
{
public:
    /// @name Constructors
    /// @{

    /// @brief Constructor.
    Model();
    /// @brief Move constructor.
    /// @param[in,out] model  The model from which to initialize this class.
    ///                       On exit, model's behavior is undefined.
    Model(Model &&model) noexcept;
    /// @}

    /// @name Operators
    /// @{

    /// @brief Move assignment.
    Model& operator=(Model &&model) noexcept;
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Destructor.
    ~Model();
    /// @}

    /// @name Initialization
    /// @{
    /// @brief Loads the HDF5 weights from file.
    /// @param[in] fileName  The HDF5 file name from which to load the weights.
    /// @param[in] verbose   If true then the dataset names and dimensions will
    ///                      be read while loading.  This is useful for
    ///                      debugging.
    /// @throws std::invalid_argument if the weights cannot be loaded from the
    ///         HDF5 file.
    void loadWeightsFromHDF5(const std::string &fileName,
                             bool verbose = false);
    /// @brief Gets the minimum number of samples in a three component
    ///        seismogram that can be handled by this network.
    /// @result The minimum requisite number of samples in a seismgoram. 
    [[nodiscard]] int getMinimumSeismogramLength() const noexcept;
    /// @brief Checks if the seismogram is a valid length.  Nominally, this
    ///        means that the seismogram length is the requisite minimum
    ///        length (\c getMinimumSeismogramLength()) and is divisible by 16.
    /// @param[in] nSamples   The number of samples.
    /// @result True indicates that the seismogram length is valid.
    [[nodiscard]] bool isValidSeismogramLength(int nSamples) const noexcept;
    /// @result True indicates that the model coefficients are set.
    [[nodiscard]] bool haveModelCoefficients() const noexcept;
    /// @brief Gets the number of input channels.  For exmaple, if this is a 
    ///        three-component picker then this will return 3.
    /// @result The number of channels expected by the picker.
    [[nodiscard]] int getInputNumberOfChannels() const noexcept;
    /// @brief Determines the number of available devices when using a system
    ///        with multiple GPU's.
    /// @result The number of available GPU's.
    [[nodiscard]] int getNumberOfDevices() const noexcept;
    /// @}

    /// @name Inference
    /// @{

    /// @brief Predicts the posterior probabilites on a seismogram.
    /// @param[in] nSamples          The numer of samples in the traces.  This
    ///                              must be greater than nSamplesInWindow.
    /// @param[in] nSamplesInWindow  The number of samples in the classification
    ///                              window.  For example, if this is 1008 then
    ///                              that is the number of points in the window
    ///                              fed to the neural network.  This must be at
    ///                              least \c getMinimumSeismogramLength() but
    ///                              cannot exceed nSamples.
    /// @param[in] vertical          The vertical channel.  This is an array
    ///                              whose dimension is [nSamples].
    /// @param[in] north             The north channel.  This is an array whose
    ///                              dimension is [nSamples].
    /// @param[in] east              The east channel.  This is an array
    ///                              whose dimension is [nSamples].
    /// @param[out] proba            The posterior probabilities.  This is an
    ///                              array whose dimension is [nSamples].
    /// @param[in] batchSize         The number of windows to process 
    ///                              simultaneously.
    void predictProbability(int nSamples,
                            int nSamplesInWindow,
                            int nCenter,
                            const float vertical[],
                            const float north[],
                            const float east[],
                            float *proba[],
                            int batchSize = 32) const;
    /// @copydoc predictProbability
    void predictProbability(int nSamples,
                            int nSamplesInWindow,
                            int nCenter,
                            const double vertical[],
                            const double north[],
                            const double east[],
                            double *proba[],
                            int batchSize = 32) const;
    /// @brief Predicts the posterior probabilities on a seismgoram.
    /// @param[in] nSamples  The number of samples in the traces.  This must
    ///                      be at least \c getMinimumSeismogramLength().
    /// @param[in] vertical  The vertical channel.  This is an array whose
    ///                      dimension is [nSamples].
    /// @param[in] north     The north channel.  This is an array whose 
    ///                      dimension is [nSamples].
    /// @param[in] east      The east channel.  This is an array whose
    ///                      dimension is [nSamples].
    /// @param[out] proba    The posterior probabilities of the phase being
    ///                      present.  This is an array whose dimension is
    ///                      [nSamples].
    /// @throws std::invalid_argument if nSamples is too small or any of the 
    ///         arrays are NULL.
    void predictProbability(int nSamples,
                            const float vertical[],
                            const float north[],
                            const float east[],
                            float *proba[]) const;
    /// @copydoc predictProbability
    void predictProbability(int nSamples,
                            const double vertical[],
                            const double north[],
                            const double east[],
                            double *proba[]) const;

    /// @brief Predicts the sample's class from the probabilities.
    /// @param[in] nSamples  The number of samples in the traces.  This must
    ///                      be at least \c getMinimumSeismogramLength().
    /// @param[in] vertical  The vertical channel.  This is an array whose
    ///                      dimension is [nSamples].
    /// @param[in] north     The north channel.  This is an array whose 
    ///                      dimension is [nSamples].
    /// @param[in] east      The east channel.  This is an array whose
    ///                      dimension is [nSamples].
    /// @param[out] clss     The class to which each simple belongs.  This is
    ///                      an array whose dimension is [nSamples].
    /// @throws std::invalid_argument if nSamples is too small or any of the
    ///         arrays are NULL.
    void predict(int nSamples,
                 const float vertical[],
                 const float north[],
                 const float east[],
                 int *clss[]) const;
    /// @copydoc predict
    void predict(int nSamples,
                 const double vertical[],
                 const double north[],
                 const double east[],
                 int *clss[]) const; 
    /// @}

    Model(const Model &model) = delete;
    Model& operator=(const Model &model) = delete;
private:
    class ModelImpl;
    std::unique_ptr<ModelImpl> pImpl; 
};
}
#endif
