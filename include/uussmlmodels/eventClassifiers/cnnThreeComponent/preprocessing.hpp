#ifndef UUSS_MLMODELS_EVENT_CLASSIFIERS_CNN_THREE_COMPONENT_PREPROCESSING_HPP
#define UUSS_MLMODELS_EVENT_CLASSIFIERS_CNN_THREE_COMPONENT_PREPROCESSING_HPP
#include <vector>
#include <tuple>
#include <memory>
namespace UUSSMLModels::EventClassifiers::CNNThreeComponent
{
/// @class Preprocessing "preprocessing.hpp" "models/eventClassifiers/cnnThreeComponent/preprocessing.hpp"
/// @brief Performs the waveform preprocessing for the S pick regressor.
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

    /// @brief Preprocesses the vertical waveforms and creates the corresponding
    ///        scalograms.
    /// @param[in] vertical      The waveform on the vertical channel.
    /// @param[in] samplingRate  The sampling rate for the signals in Hz.
    /// @result The amplitude scalogram for the vertical channel.
    ///         This is an  [getScalogramLength() x getNumberOfScales()] matrix
    ///         stored in row major order.
    template<typename U>
    [[nodiscard]] std::vector<U> process(const std::vector<U> &vertical,
                                          double samplingRate = 100);

    /// @brief Preprocesses the vertical, north, and east waveforms and creates
    ///        the corresponding scalograms.
    /// @param[in] vertical      The waveform on the vertical channel.
    /// @param[in] north         The waveform on the north channel.
    /// @param[in] east          The waveform on the east channel.
    /// @param[in] samplingRate  The sampling rate for the signals in Hz.
    /// @result The amplitude scalograms for the vertical, north, and
    ///         east waveforms.  These are each
    ///         [getScalogramLength() x getNumberOfScales()] matrices
    ///         stored in row major order.
    template<typename U>
    [[nodiscard]] std::tuple<std::vector<U>, std::vector<U>, std::vector<U>>
        process(const std::vector<U> &vertical,
                const std::vector<U> &north,
                const std::vector<U> &east,
                double samplingRate = 100);
    /// @}

    /// @name Target Sampling Rate
    /// @{

    /// @result The sampling rate of the scalogram in Hz.
    [[nodiscard]] static double getScalogramSamplingRate() noexcept;
    /// @result The sampling period of the scalogram in seconds.
    [[nodiscard]] static double getScalogramSamplingPeriod() noexcept;
    /// @}

    /// @name Number of Windows
    /// @{

    /// @result The number of time samples in the scsalogram.
    [[nodiscard]] static int getScalogramLength() noexcept;
    /// @result The number of scales in each transform.
    [[nodiscard]] static int getNumberOfScales() noexcept; 
    /// @result The center frequencies of each in Hz.
    [[nodiscard]] std::vector<double> getCenterFrequencies() const; 
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
