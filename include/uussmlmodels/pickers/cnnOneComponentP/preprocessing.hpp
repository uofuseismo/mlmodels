#ifndef UUSS_MLMODELS_PICKERS_CNN_ONE_COMPONENT_P_PREPROCESSING_HPP
#define UUSS_MLMODELS_PICKERS_CNN_ONE_COMPONENT_P_PREPROCESSING_HPP
#include <vector>
#include <memory>
namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP
{
/// @class Preprocessing "preprocessing.hpp" "models/pickers/cnnOneCompoentP/preprocessing.hpp"
/// @brief Performs the waveform preprocessing for the P pick regressor.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class Preprocessing
{
public:
    /// @name Constructors
    /// @{

    /// @brief Constructor.
    Preprocessing();
    /// @brief Move constructor.
    /// @param[in,out] preprocess  The class from which to initialize this
    ///                            class.  On exit, preprocess's behavior is
    ///                            undefined.
    Preprocessing(Preprocessing &&process) noexcept;
    /// @}

    /// @name Destructors
    /// @{

    /// @brief Resets class and releases space.
    void clear() noexcept;
    /// @brief Destructor.
    ~Preprocessing();
    /// @}

    /// @name Process Waveforms
    /// @{

    /// @brief Preprocesses the vertical waveform.
    /// @param[in] vertical      The waveform on the vertical channel.
    /// @param[in] samplingRate  The sampling rate for the signals in Hz.
    /// @result The processed wavefrorm on the vertical channel.
    template<typename U>
    [[nodiscard]] std::vector<U> process(const std::vector<U> &vertical,
                                         double samplingRate = 100);
               
    /// @}

    /// @name Target Sampling Rate
    /// @{

    /// @result The sampling rate of the processed signals in Hz.
    [[nodiscard]] static double getTargetSamplingRate() noexcept;
    /// @result The sampling period of the processed signals in seconds.
    [[nodiscard]] static double getTargetSamplingPeriod() noexcept;
    /// @}

    /// @name Operators
    /// @{

    /// @brief Move assignment operator.
    /// @param[in,out] preprocess  The class whose memory will be moved to this.
    ///                            On exit, preprocess's behavior is undefined.
    /// @result The memory from preprocess moved to this.
    Preprocessing& operator=(Preprocessing &&preprocess) noexcept;
    /// @}

    Preprocessing(const Preprocessing &) = delete;
    Preprocessing& operator=(const Preprocessing &) = delete;
private:
    class PreprocessingImpl;
    std::unique_ptr<PreprocessingImpl> pImpl;
};
}
#endif
