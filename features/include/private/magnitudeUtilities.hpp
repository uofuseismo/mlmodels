#ifndef PRIVATE_MAGNITUDEUTILITIES_HPP
#define PRIVATE_MAGNITUDEUTILITIES_HPP
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <rtseis/transforms/continuousWavelet.hpp>
#include <rtseis/transforms/wavelets/morlet.hpp>
#include <rtseis/filterImplementations/detrend.hpp>
#include <rtseis/filterImplementations/taper.hpp>
#include <rtseis/filterImplementations/iirFilter.hpp>
#include <rtseis/filterImplementations/sosFilter.hpp>
#include <rtseis/filterRepresentations/ba.hpp>
#include <rtseis/filterRepresentations/sos.hpp>
#include <rtseis/filterDesign/iir.hpp>
#include <rtseis/utilities/interpolation/weightedAverageSlopes.hpp>
#include "private/distanceAzimuth.hpp"
namespace
{
template<typename T>
[[nodiscard]] T mean(const int n, const T *__restrict__ x)
{
    if (n == 0){return 0;}
    double xSum = 0;
    for (int i = 0; i < n; ++i)
    {
        xSum = xSum + x[i];
    }
    return static_cast<T> (xSum/n);
}
template<typename T>
[[nodiscard]] T variance(const int n, const T *__restrict__ x)
{
    if (n < 2){return 0;}
    auto xMean = mean(n, x);
    double xSum2 = 0;
    for (int i = 0; i < n; ++i)
    {
        auto resid = x[i] - xMean;
        xSum2 = xSum2 + static_cast<double> (resid*resid);
    }
    return static_cast<T> (xSum2/(n - 1));
}
template<typename T>
[[nodiscard]] T absoluteMaximum(const int n, const T *__restrict__ x)
{
    T result = 0;
    for (int i = 0; i < n; ++i)
    {
        result = std::max(result, std::abs(x[i]));
    }
    return result;
}
template<typename T>
[[nodiscard]] T absoluteMaximum(const std::vector<T> &x)
{
    return absoluteMaximum(x.size(), x.data());
}
/// Convenience interface for computing distance and azimuth.
void distanceAzimuth(const UUSS::Features::Magnitude::Hypocenter &hypocenter,
                     const UUSS::Features::Magnitude::Channel &channel,
                     double *distance, double *backAzimuth)
{
    auto sourceLatitude    = hypocenter.getLatitude();
    auto sourceLongitude   = hypocenter.getLongitude();
    auto receiverLatitude  = channel.getLatitude();
    auto receiverLongitude = channel.getLongitude();
    double greatCircleDistance, azimuth;
    computeDistanceAzimuthWGS84(sourceLatitude, sourceLongitude,
                                receiverLatitude, receiverLongitude,
                                &greatCircleDistance,
                                distance,
                                &azimuth,
                                backAzimuth);
}
/// Gets the dominant and frequency from the CWT. 
std::pair<double, double> getDominantFrequencyAndAmplitude(
    const int nScales, const int nSamples,
    const int iStart, const int iEnd,
    const double *__restrict__ centerFrequencies,
    const double *__restrict__ amplitudeCWT)
{
    std::pair<double, double> dominantFrequencyAmplitude{0, 0};
    for (int j = 0; j < nScales; ++j)
    {
        for (int i = iStart; i < iEnd; ++i)
        {
            auto indx = j*nSamples + i;
            auto amplitude = amplitudeCWT[indx];
            if (amplitude > dominantFrequencyAmplitude.second)
            {
                dominantFrequencyAmplitude.first  = centerFrequencies[j];
                dominantFrequencyAmplitude.second = amplitude;
            }
        }
    }
    return dominantFrequencyAmplitude;
}
/// Average amplitude in the frequency in the bins
std::vector<std::pair<double, double>> getAverageFrequencyAndAmplitude(
    const int nScales, const int nSamples,
    const int iStart, const int iEnd,
    const double *__restrict__ centerFrequencies,
    const double *__restrict__ amplitudeCWT)
{
    std::vector<std::pair<double, double>> averageFrequencyAmplitude(nScales);
    for (int j = 0; j < nScales; ++j)
    {
        double averageAmplitude = 0;
        for (int i = iStart; i < iEnd; ++i)
        {
            auto indx = j*nSamples + i;
            averageAmplitude = averageAmplitude + amplitudeCWT[indx];
        }
        averageFrequencyAmplitude[j] = std::pair(centerFrequencies[j],
                                                 averageAmplitude/nScales);
    }
    return averageFrequencyAmplitude;
}
/// Compute CWT
void computeVelocityScalogram(
    const std::vector<double> &velocitySignal,
    std::vector<double> *amplitudeCWT,
    RTSeis::Transforms::ContinuousWavelet<double> &cwt)
{   
    // Calculate the scalogram
    cwt.transform(velocitySignal.size(), velocitySignal.data());
    auto nScales  = cwt.getNumberOfScales(); 
    auto nSamples = cwt.getNumberOfSamples();
    if (static_cast<int> (amplitudeCWT->size()) != nScales*nSamples)
    {
        amplitudeCWT->resize(nScales*nSamples, 0);
    }
    auto aPtr = amplitudeCWT->data();
    cwt.getAmplitudeTransform(nSamples, nScales, &aPtr);
}
/// Extract domain domain features
[[nodiscard]]
UUSS::Features::Magnitude::TemporalFeatures
    getTimeDomainFeatures(const int i0, const int i1,
                          const std::vector<double> &velocitySignal)
{
    UUSS::Features::Magnitude::TemporalFeatures temporalFeatures;
    // (1) The variance is the signal power minus the DC power.
    auto nSamples = i1 - i0;
    auto varianceNoise = variance(nSamples, velocitySignal.data() + i0);
    // Get difference of min/max amplitude 
    const auto [vMinNoise,  vMaxNoise]
        = std::minmax_element(velocitySignal.data() + i0,
                              velocitySignal.data() + i1);
    temporalFeatures.setVariance(varianceNoise);
    temporalFeatures.setMinimumAndMaximumValue(
            std::pair(*vMinNoise, *vMaxNoise));
    return temporalFeatures;
}
/// Extract spectral domain features
[[nodiscard]]
UUSS::Features::Magnitude::SpectralFeatures
    getSpectralDomainFeatures(const int nSamples, const int nScales, 
                              const int i0, const int i1,
                              const std::vector<double> &centerFrequencies,
                              const std::vector<double> &amplitudeCWT)
{
     UUSS::Features::Magnitude::SpectralFeatures spectralFeatures;
     auto dominantFrequencyAmplitude
        = getDominantFrequencyAndAmplitude(nScales, nSamples,
                                           i0, i1,
                                           centerFrequencies.data(),
                                           amplitudeCWT.data());
     spectralFeatures.setDominantFrequencyAndAmplitude(
         dominantFrequencyAmplitude);

     auto averageFrequencyAmplitude
        = getAverageFrequencyAndAmplitude(nScales, nSamples,
                                          i0, i1,
                                          centerFrequencies.data(),
                                          amplitudeCWT.data());
     spectralFeatures.setAverageFrequenciesAndAmplitudes(
         averageFrequencyAmplitude);
 
     return spectralFeatures;
}

/*
[[maybe_unused]]
void processAcceleration(
    const std::vector<double> &signal,
    std::vector<double> *velocitySignal,
    std::vector<double> *workSpace,
    const bool isAcceleration,
    RTSeis::FilterImplementations::Taper<double> &window,
    RTSeis::FilterImplementations::SOSFilter<RTSeis::ProcessingMode::POST, double> &iirHighPass,
    RTSeis::FilterImplementations::IIRFilter<RTSeis::ProcessingMode::POST, double> &iirIntegrator)
{
    if (!isAcceleration){return;} // Nothing to do
    int nSamples = static_cast<int> (signal.size());
    if (workSpace->size() != signal.size())
    {
        workSpace->resize(signal.size());
    }
    std::copy(signal.begin(), signal.end(), workSpace->begin());
    auto yWorkPtr = velocitySignal->data();
    // Remove mean
    RTSeis::FilterImplementations::Detrend<double>
        demean(RTSeis::FilterImplementations::DetrendType::Constant);
    demean.apply(nSamples, workSpace->data(), &yWorkPtr);
    std::copy(yWorkPtr, yWorkPtr + nSamples, workSpace->begin());
    // Window
    window.apply(nSamples, workSpace->data(), &yWorkPtr);
    std::copy(yWorkPtr, yWorkPtr + nSamples, workSpace->begin());
    // Highpass filter
    iirHighPass.apply(nSamples, workSpace->data(), &yWorkPtr);
    std::copy(yWorkPtr, yWorkPtr + nSamples, workSpace->begin());
    // Integrate
    iirIntegrator.apply(nSamples, workSpace->data(), &yWorkPtr);
}

[[maybe_unused]]
void processVelocity(
    const std::vector<double> &signal,
    std::vector<double> *velocitySignal,
    std::vector<double> *workSpace,
    const double simpleResponse,
    const double maxPeakGroundVelocity,
    const bool isAcceleration,
    RTSeis::FilterImplementations::Taper<double> &window,
    RTSeis::FilterImplementations::SOSFilter<RTSeis::ProcessingMode::POST, double> &iirVelocityFilter)
{
    int nSamples = static_cast<int> (signal.size());
    if (workSpace->size() < std::max(signal.size(), velocitySignal->size()))
    {
        workSpace->resize(std::max(signal.size(), velocitySignal->size()));
    }
    // Input acceleration seismograms were copied to the velocity signal
    // as a temporary input.
    if (isAcceleration)
    {
        std::copy(velocitySignal->begin(), velocitySignal->end(),
                  workSpace->begin());
    }
    else
    {
        // Otherwise we copy to the workspace directly from the input signal.
        std::copy(signal.begin(), signal.end(), workSpace->begin());
    }
    auto yWorkPtr = velocitySignal->data();
    // Remove mean
    RTSeis::FilterImplementations::Detrend<double>
        demean(RTSeis::FilterImplementations::DetrendType::Constant);
    demean.apply(nSamples, workSpace->data(), &yWorkPtr);
    std::copy(yWorkPtr, yWorkPtr + nSamples, workSpace->begin());
    // Window
    window.apply(nSamples, workSpace->data(), &yWorkPtr); 
    std::copy(yWorkPtr, yWorkPtr + nSamples, workSpace->begin());
    // Remove gain
    if (!isAcceleration)
    {
        double gainInverse = 1./simpleResponse;
        std::transform(workSpace->begin(), workSpace->end(),
                       workSpace->begin(),
                       [gainInverse](double x)
                       {
                           return gainInverse*x;
                       });
    }
    // High-pass filter
    iirVelocityFilter.apply(nSamples, workSpace->data(), &yWorkPtr);
    // Check the gain is reasonable
    auto pgvMax = absoluteMaximum(*velocitySignal);
    if (pgvMax > maxPeakGroundVelocity)
    {
        throw std::invalid_argument("Max PGV = "
                 + std::to_string(pgvMax*1.e-4) + " cm/s exceeds "
                 + std::to_string(maxPeakGroundVelocity*1.e-4)
                 + " cm/s - check response.");
    }
}
*/
}
#endif
