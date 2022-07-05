#ifndef UUSS_FEATURES_MAGNITUDE_SPECTRALFEATURES_HPP
#define UUSS_FEATURES_MAGNITUDE_SPECTRALFEATURES_HPP
#include <memory>
#include <vector>
namespace UUSS::Features::Magnitude
{
/// @class SpectralFeatures "spectralFeatures.hpp" "uuss/features/magnitude/spectralFeatures.hpp"
/// @brief Features measured on the frequency/scale domain signal representation.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class SpectralFeatures
{
public:
    /// @brief Constructor.
    SpectralFeatures();
    /// @brief Copy constructor.
    SpectralFeatures(const SpectralFeatures &features);
    /// @brief Move constructor.
    SpectralFeatures(SpectralFeatures &&features) noexcept;

    /// @brief Copy assignment.
    SpectralFeatures& operator=(const SpectralFeatures &features);
    /// @breif Move assignment.
    SpectralFeatures& operator=(SpectralFeatures &&features) noexcept;

    /// @brief Sets the dominant frequency and corresponding amplitude.
    /// @param[in] frequencyAmplitude  The frequencyAmplitude.first is the
    ///                                frequency in Hz corresponding to the
    ///                                dominant amplitude while
    ///                                frequencyAmplitude.second is
    ///                               the amplitude.
    /// @throws std::invalid_argument if either the frequency is not positive
    ///         or the amplitude is negative.
    void setDominantFrequencyAndAmplitude(const std::pair<double, double> &frequencyAmplitude);
    /// @result result.first is the frequency in Hz corresponding to the 
    ///         amplitude and result.second is the amplitude of the dominant
    ///         frequency.
    /// @throws std::runtime_error if \c haveDominantFrequencyAndAmplitude()
    ///         is false.
    [[nodiscard]] std::pair<double, double> getDominantFrequencyAndAmplitude() const;
    /// @result True indicates the dominant frequency and amplitude were set.
    [[nodiscard]] bool haveDominantFrequencyAndAmplitude() const noexcept; 


    /// @brief Sets the average amplitude at a frequency over a processing window.
    void setAverageFrequenciesAndAmplitudes(const std::vector<std::pair<double, double>> &amplitudes);
    [[nodiscard]] std::vector<std::pair<double, double>> getAverageFrequenciesAndAmplitudes() const;
    [[nodiscard]] bool haveAverageFrequenciesAndAmplitudes() const noexcept;
/*
    /// @brief Sets the dominant frequency and corresponding cumulative amplitude.
    /// @param[in] periodCumAmp  The periodCumAmp.first is the period in
    ///                          seconds corresponding to the dominant
    ///                          cumulative amplitude while periodCumAmp.second
    ///                          is the cumulative amplitude.
    /// @throws std::invalid_argument if either the period is not positive
    ///         or the amplitude is negative.
    void setDominantPeriodAndCumulativeAmplitude(const std::pair<double, double> &periodCumAmp);
    /// @result result.first is the period in seconds corresponding to the 
    ///         dominant cumulative amplitude and result.second is the
    ///         cumulative amplitude at that period.
    /// @throws std::runtime_error if \c haveDominantPeriodAndCumulativeAmplitude()
    ///         is false.
    std::pair<double, double> getDominantPeriodAndCumulativeAmplitude() const;
    /// @result True indicates the dominant cumulative energy and corresponding
    ///         period wer set.
    [[nodiscard]] bool haveDominantPeriodAndCumulativeAmplitude() const noexcept;
*/

    /// @brief Resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~SpectralFeatures();
private:
    class SpectralFeaturesImpl;
    std::unique_ptr<SpectralFeaturesImpl> pImpl;
};
}
#endif
