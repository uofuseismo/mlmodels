#ifndef UUSS_FEATURES_MAGNITUDE_SPECTRALFEATURES_HPP
#define UUSS_FEATURES_MAGNITUDE_SPECTRALFEATURES_HPP
#include <memory>
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

    /// @brief Sets the dominant period and corresponding amplitude.
    /// @param[in] periodAmplitude  The periodAmplitude.first is the period in
    ///                             seconds corresponding to the dominant
    ///                             amplitude while periodAmplitude.second is
    ///                             the amplitude.
    /// @throws std::invalid_argument if either the period is not positive
    ///         or the amplitude is negative.
    void setDominantPeriodAndAmplitude(const std::pair<double, double> &periodAmplitude);
    /// @result result.first is the period in seconds corresponding to the 
    ///         amplitude and result.second is the amplitude of the dominant
    ///         period.
    /// @throws std::runtime_error if \c haveDominantPeriodAndAmplitude()
    ///         is false.
    [[nodiscard]] std::pair<double, double> getDominantPeriodAndAmplitude() const;
    /// @result True indicates the dominant period and amplitude were set.
    [[nodiscard]] bool haveDominantPeriodAndAmplitude() const noexcept; 

    /// @brief Sets the dominant period and corresponding cumulative amplitude.
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
