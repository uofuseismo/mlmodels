#ifndef UUSS_FEATURES_MAGNITUDE_TEMPORALFEATURES_HPP
#define UUSS_FEATURES_MAGNITUDE_TEMPORALFEATURES_HPP
#include <memory>
namespace UUSS::Features::Magnitude
{
/// @class TemporalFeatures "temporalFeatures.hpp" "uuss/features/magnitude/temporalFeatures.hpp"
/// @brief Features measured on the time domain signal representation.
/// @copyright Ben Baker (University of Utah) distributed under the MIT license.
class TemporalFeatures
{
public:
    /// @brief Constructor.
    TemporalFeatures();
    /// @brief Copy constructor.
    TemporalFeatures(const TemporalFeatures &features);
    /// @brief Move constructor.
    TemporalFeatures(TemporalFeatures &&features) noexcept;

    /// @brief Copy assignment.
    TemporalFeatures& operator=(const TemporalFeatures &features);
    /// @breif Move assignment.
    TemporalFeatures& operator=(TemporalFeatures &&features) noexcept;

    /// @brief Sets the variance in the signal.
    /// @param[in] variance  The signal variance in (micrometers/second)^2.
    /// @throws std::invalid_argument if the variance is negative.
    /// @note This likely has units of (micrometers/second)^2.
    void setVariance(double variance);
    /// @result The variance in the signal.
    [[nodiscard]] double getVariance() const;
    /// @result True indicates the signal variance was set.
    [[nodiscard]] bool haveVariance() const noexcept;

    /// @brief The minimum and maximum value of the signal.
    /// @param[in] minMax  The minimum and maximum value of the signal.
    ///                    This should be in units of micrometers per second.
    /// @throws std::invalid_argument if minMax.first > minMax.second. 
    void setMinimumAndMaximumValue(const std::pair<double, double> &minMax);
    /// @result The minimum and maximum of the signal in micrometers per second.
    /// @throws std::runtime_error if \c haveMinimumAndMaximumValue() is false.
    [[nodiscard]] std::pair<double, double> getMinimumAndMaximumValue() const;
    /// @result True indicates the minimum and maximum value was set.
    [[nodiscard]] bool haveMinimumAndMaximumValue() const noexcept;

    /// @brief Resets the class.
    void clear() noexcept;
    /// @brief Destructor.
    ~TemporalFeatures();
private:
    class TemporalFeaturesImpl;
    std::unique_ptr<TemporalFeaturesImpl> pImpl;
};
}
#endif
