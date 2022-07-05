#ifndef UUSS_FEATURES_MAGNITUDE_CHANNELFEATURES_HPP
#define UUSS_FEATURES_MAGNITUDE_CHANNELFEATURES_HPP
#include <memory>
namespace UUSS::Features::Magnitude
{
class Hypocenter;
class Channel;
class TemporalFeatures;
class SpectralFeatures;
}
namespace UUSS::Features::Magnitude
{
/*
*/
/// @class ChannelFeatures "channelFeatures.hpp" "uuss/features/magnitude/channelFeatures.hpp"
/// @brief Extracts features for computing a magnitude from a single channel.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class ChannelFeatures
{
public:
    /// @brief Constructor.
    ChannelFeatures();

    /// @result The sampling rate of the signal from which the features
    ///         will be extracted in Hz.
    [[nodiscard]] static double getTargetSamplingRate() noexcept;
    /// @result The sampling period of the signal from which the features
    ///         will be extracted in seconds.
    [[nodiscard]] static double getTargetSamplingPeriod() noexcept;
    /// @result The duration of the signal from which the features will
    ///         be extracted in seconds.
    [[nodiscard]] static double getTargetSignalDuration() noexcept;
    /// @result The length of the signal from which the features will
    ///         be extracted in seconds.
    [[nodiscard]] static int getTargetSignalLength() noexcept;
    /// @result The number of seconds before and after the arrival where the 
    ///         processing will be performed.  For example,
    ///         arrivalTime + result.first will indicate where the processing
    ///         window begins while arrivalTime + result.second will indicate
    ///         where the processing will end.
    [[nodiscard]] static std::pair<double, double> getArrivalTimeProcessingWindow() noexcept;

    /// @name Step 1: Initialization
    /// @{

    /// @brief Initializes the feature extraction tool for this channel.
    /// @param[in] channel  The channel information.
    /// @throws std::invalid_argument if the channel.haveSamplingRate() or
    ///         channel.haveSimpleResponse() is false.
    /// @note Initialization is expensive.  Do this as infrequently as possible.
    void initialize(const Channel &channel);
    /// @result True indicates the class is initialized.
    [[nodiscard]] bool isInitialized() const noexcept;
    /// @result A reference to the channel information.
    /// @throws std::runtime_error if the class is not initialized.
    [[nodiscard]] Channel& getChannelReference() const;

    /// @result The sampling rate in Hz.
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] double getSamplingRate() const;
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] std::string getSimpleResponseUnits() const;
    /// @result The simple response value.  When the input signal is divided
    ///         by this number then the result have units of m/s or m/s^2 as
    ///         indicated by \c getSimpleResponseUnits().
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] double getSimpleResponseValue() const;
    /// @}

    /// @brief Sets the hypocenter to which this signal corresponds.
    /// @param[in] hypocenter  The hypocenter information.
    /// @throws std::runtime_error if \c isInitialized() is false.
    /// @throws std::invalid_argument if the \c hypocenter.haveLatitude() or
    ///         \c hypocenter.haveLongitude() is false.
    void setHypocenter(const Hypocenter &hypocenter);
    /// @result True indicates the hypocenter was set.
    [[nodiscard]] bool haveHypocenter() const noexcept;
    /// @result The source-receiver distance in kilometers.
    /// @throws std::runtime_error if \c haveHypocenter() or 
    [[nodiscard]] double getSourceReceiverDistance() const;

    /// @brief Processes the signal.
    /// @param[in] signal   The signal to process.
    /// @param[in] arrivalTimeRelativeToStart  The phase arrival time in seconds
    ///                                        relative to the signal start.
    /// @throws std::runtime_error if \c isInitialized() is false.
    /// @throws std::invalid_argument if the signal is too small or the arrival
    ///         time relative to the start is less than the processing window
    ///         start time.
    /// @sa \c getArrivalTimeProcessingWindow(), \c getTargetSignalDuration()
    template<typename U> void process(const std::vector<U> &signal,
                                      double arrivalTimeRelativeToStart);
    template<typename U> void process(int n, const U signal[],
                                      double arrivalTimeRelativeToStart);
    /// @result True indicates the signal was set and the features
    ///         were extracted.
    [[nodiscard]] bool haveFeatures() const noexcept;

    /// @result The velocity signal from which to extract features.
    /// @throws std::runtime_error if \c haveSignal() is false.
    [[nodiscard]] std::vector<double> getVelocitySignal() const;


    void clear() noexcept; 
    ~ChannelFeatures();


    ChannelFeatures(const ChannelFeatures &) = delete;
    ChannelFeatures(ChannelFeatures &&) noexcept = delete;
    ChannelFeatures& operator=(const ChannelFeatures &) = delete;
    ChannelFeatures& operator=(ChannelFeatures &&) noexcept = delete;
private:
    class FeaturesImpl;
    std::unique_ptr<FeaturesImpl> pImpl;
};
}
#endif