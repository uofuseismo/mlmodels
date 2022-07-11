#ifndef UUSS_FEATURES_MAGNITUDE_SFEATURES_HPP
#define UUSS_FEATURES_MAGNITUDE_SFEATURES_HPP
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
/// @class SFeatures "sFeatures.hpp" "uuss/features/magnitude/sFeatures.hpp"
/// @brief Extracts features for computing a magnitude from the S arrival.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class SFeatures
{
public:
    SFeatures();

    /// @result The sampling rate of the signal from which the features
    ///         will be extracted in Hz.
    [[nodiscard]] static double getTargetSamplingRate() noexcept;
    /// @result The sampling period of the signal from which the features
    ///         will be extracted in seconds.
    [[nodiscard]] static double getTargetSamplingPeriod() noexcept;
    /// @result The duration of the signal from which the features will
    ///         be extracted in seconds.
    [[nodiscard]] double getTargetSignalDuration() const;
    /// @result The length of the signal from which the features will
    ///         be extracted in seconds.
    [[nodiscard]] int getTargetSignalLength() const;
    /// @result The number of seconds before and after the arrival where the 
    ///         processing will be performed.  For example,
    ///         arrivalTime + result.first will indicate where the processing
    ///         window begins while arrivalTime + result.second will indicate
    ///         where the processing will end.
    [[nodiscard]] std::pair<double, double> getArrivalTimeProcessingWindow() const noexcept;

    /// @name Step 1: Initialization
    /// @{

    /// @brief Initializes the feature extraction tool for this channel.
    /// @param[in] northChannel  The north (1) channel information.
    /// @param[in] eastChannel   The east (2) channel information.
    /// @throws std::invalid_argument if the channel.haveSamplingRate() or
    ///         channel.haveSimpleResponse() is false.
    /// @note Initialization is expensive.  Do this as infrequently as possible.
    void initialize(const Channel &northChannel,
                    const Channel &eastChannel);
    /// @result True indicates the class is initialized.
    [[nodiscard]] bool isInitialized() const noexcept;

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
    /// @result The hypocenter.
    /// @throws std::runtime_error if \c haveHypocenter() is false.
    [[nodiscard]] Hypocenter getHypocenter() const;
    /// @result True indicates the hypocenter was set.
    [[nodiscard]] bool haveHypocenter() const noexcept;

    /// @brief Processes the signal.  
    /// @param[in] nSignal   The north (or 1 channel) signal to process.
    /// @param[in] eSignal   The east (or 2 channel) signal to process.
    /// @param[in] arrivalTimeRelativeToStart  The phase arrival time in seconds
    ///                                        relative to the starts of all the
    ///                                        signal.
    /// @throws std::runtime_error if \c isInitialized() is false or 
    ///         \c haveHypocenter() is false.
    /// @throws std::invalid_argument if the signal is too small or the arrival
    ///         time relative to the start is less than the processing window
    ///         start time.  Additionally, this will throw if not all the
    ///         signals are of the same length.
    /// @sa \c getArrivalTimeProcessingWindow(), \c getTargetSignalDuration()
    void process(const std::vector<double> &nSignal,
                 const std::vector<double> &eSignal,
                 double arrivalTimeRelativeToStart);
    void process(int nSamples,
                 const double *nSignal, const double *eSignal,
                 double arrivalTimeRelativeToStart);
    /// @result True indicates the input signal was processed and the
    ///         velocity-based features are available.
    [[nodiscard]] bool haveFeatures() const noexcept;
    /// @result The temporal features computed on the radial channel for the
    ///         the pre-arrival noise.
    [[nodiscard]] TemporalFeatures getRadialTemporalNoiseFeatures() const;
    /// @result The temporal features computed on the radial channel for the
    ///         the S signal.
    [[nodiscard]] TemporalFeatures getRadialTemporalSignalFeatures() const;
    /// @result The temporal features computed on the transverse channel for the
    ///         the pre-arrival noise.
    [[nodiscard]] TemporalFeatures getTransverseTemporalNoiseFeatures() const;
    /// @result The temporal features computed on the transverse channel for the
    ///         the S signal.
    [[nodiscard]] TemporalFeatures getTransverseTemporalSignalFeatures() const;
    /// @result The spectral features computed on the radial channel for
    ///         the pre-arrival noise.
    [[nodiscard]] SpectralFeatures getRadialSpectralNoiseFeatures() const;
    /// @result The spectral features computed on the radial channel for
    ///         the S signal signal.
    [[nodiscard]] SpectralFeatures getRadialSpectralSignalFeatures() const;
    /// @result The spectral features computed on the transverse channel for
    ///         the pre-arrival noise.
    [[nodiscard]] SpectralFeatures getTransverseSpectralNoiseFeatures() const;
    /// @result The spectral features computed on the transverse channel for
    ///         the S signal signal.
    [[nodiscard]] SpectralFeatures getTransverseSpectralSignalFeatures() const;
    /// @result The source depth. 
    [[nodiscard]] double getSourceDepth() const;
    /// @result The source-receiver distance in kilometers.
    [[nodiscard]] double getSourceReceiverDistance() const;
    /// @result The back-azimuth in degrees measured positive east of north.
    /// @throws std::runtime_error if \c haveHypocenter() is false.
    [[nodiscard]] double getBackAzimuth() const;

    /// @result The velocity signal from which to extract features.
    /// @throws std::runtime_error if \c haveFeatures() is false.
    //[[nodiscard]] std::vector<double> getVerticalVelocitySignal() const;
    [[nodiscard]] std::vector<double> getRadialVelocitySignal() const;
    [[nodiscard]] std::vector<double> getTransverseVelocitySignal() const;


    void clear() noexcept; 
    ~SFeatures();

private:
    class FeaturesImpl;
    std::unique_ptr<FeaturesImpl> pImpl;
};
}
#endif
