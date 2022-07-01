#ifndef UUSS_FEATURES_MAGNITUDE_VERTICALCHANNELFEATURES_HPP
#define UUSS_FEATURES_MAGNITUDE_VERTICALCHANNELFEATURES_HPP
#include <memory>
namespace UUSS::Features::Magnitude
{
class Hypocenter;
class ChannelDetails;
}
namespace UUSS::Features::Magnitude
{
/// @class VerticalChannelFeatures "verticalChannelFeatures.hpp" "uuss/features/verticalChannelFeatures.hpp"
/// @brief Extracts features for computing a magnitude from a single channel
///        station.
class VerticalChannelFeatures
{
public:
    VerticalChannelFeatures();

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

    /// @name Initialization
    /// @{

    /// @brief Initializes the feature extraction tool.
    /// @param[in] samplingRate    The sampling rate in Hz.
    /// @param[in] simpleResponse  The simple response.  Division by this
    ///                            number will result in the signal being in
    ///                            the units specified by units.
    /// @param[in] units           The units.  This is from the AQMS database. 
    ///                            This can be DU/M/S for velocity or DU/M/S**2
    ///                            for acceleration.
    /// @throws std::invalid_argument if the sampling rate is not positive,
    ///         the simple response is 0, or the units are no thandled.
    void initialize(double samplingRate,
                    double simpleResponse,
                    const std::string &units);
    /// @result True indicates the class is initialized.
    [[nodiscard]] bool isInitialized() const noexcept;

    /// @result The sampling rate in Hz.
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] double getSamplingRate() const;
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] std::string getSimpleResponseUnits() const;
    /// @result The simple response.
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] double getSimpleResponse() const;
    /// @}

    /// @brief Processes the signal.
    /// @throws std::runtime_error if \c isInitialized() is false.
    /// @throws std::invalid_argument if the signal is too small or the arrival
    ///         time relative to the start is less than the processing window
    ///         start time.
    /// @sa \c getArrivalTimeProcessingWindow(), \c getTargetSignalDuration()
    template<typename U> void process(const std::vector<U> &signal,
                                      double arrivalTimeRelativeToStart);
    template<typename U> void process(int n, const U signal[],
                                      double arrivalTimeRelativeToStart);
    [[nodiscard]] bool haveSignal() const noexcept;

    /// @result The velocity signal from which to extract features.
    /// @throws std::runtime_error if \c haveSignal() is false.
    [[nodiscard]] std::vector<double> getVelocitySignal() const;


    void clear() noexcept; 
    ~VerticalChannelFeatures();

private:
    class FeaturesImpl;
    std::unique_ptr<FeaturesImpl> pImpl;
};
}
#endif
