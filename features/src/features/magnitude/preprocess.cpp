#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
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
#include "uuss/features/magnitude/preprocess.hpp"

using namespace UUSS::Features::Magnitude;
namespace RFilters = RTSeis::FilterImplementations;
namespace RFilterDesign = RTSeis::FilterDesign;

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100

namespace
{
/// @brief Performs resampling with Wiggins interpolation.
template<typename U>
std::vector<double> resample(const int n,
                             const U *__restrict__ y,
                             const double samplingRate,
                             const double targetSamplingRate = 100)
{
    if (n < 2){throw std::runtime_error("Array must be at least 2");}
    if (samplingRate <= 0)
    {   
        throw std::invalid_argument("Sampling rate must be positive");
    }   
    auto samplingPeriod = 1./samplingRate;
    if (targetSamplingRate <= 0)
    {   
        throw std::invalid_argument("Target sampling rate must be positive");
    }   
    auto targetSamplingPeriod = 1./targetSamplingRate;
    // Initialize interpolator
    std::vector<double> yWork(n);
    std::copy(y, y + n, yWork.begin());
    std::pair<double, double> xLimits{0, (n - 1)*samplingPeriod};
    RTSeis::Utilities::Interpolation::WeightedAverageSlopes<double> wiggins; 
    wiggins.initialize(yWork.size(), xLimits, yWork.data());
    // Create interpolation points
    auto tMax = xLimits.second; 
    auto nNewSamples
        = static_cast<int> (std::round(tMax/targetSamplingPeriod)) + 1;
    while ((nNewSamples - 1)*targetSamplingPeriod > tMax && nNewSamples > 0)
    {
        nNewSamples = nNewSamples - 1;
    }   
    std::pair<double, double>
         newInterval{0, (nNewSamples - 1)*targetSamplingPeriod};
    std::vector<double> result(nNewSamples);
    auto yPtr = result.data(); 
    wiggins.interpolate(nNewSamples, newInterval, &yPtr);
    return result;
}
/// Absolute maximum of signal
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
/// Absolute maximum of signal
template<typename T>
[[nodiscard]] T absoluteMaximum(const std::vector<T> &x) 
{
    return absoluteMaximum(x.size(), x.data());
}
}

class Preprocess::PreprocessImpl
{
public:
    PreprocessImpl()
    {
        const double pct = 5;
        mWindow.initialize(pct, RFilters::TaperWindowType::Hann);
    }
    /// Create the velocity filter
    void createVelocityFilter()
    {
        const double nyquistFrequency
            = 0.5*Preprocess::getTargetSamplingRate(); //mTargetSamplingRate;
        auto normalizedLowCorner  = 1/nyquistFrequency;
        auto normalizedHighCorner = std::min(nyquistFrequency*0.9,
                                             18/nyquistFrequency);
        const std::vector<double> Wn{normalizedLowCorner,
                                     normalizedHighCorner};
        const int order{3}; // Will be doubled in design
        const double rp{20};
        const double rs{5};
        constexpr auto band = RFilterDesign::Bandtype::BANDPASS;
        constexpr auto prototype = RFilterDesign::IIRPrototype::BUTTERWORTH;
        constexpr auto digital = RFilterDesign::IIRFilterDomain::DIGITAL;
        mVelocityFilter = RFilterDesign::IIR::designSOSIIRFilter(
                 order, Wn.data(), rp, rs, band, prototype, digital);
        // Initialize velocity filter
        mIIRVelocityFilter.initialize(mVelocityFilter);
    }
    /// Create the acceleration filter
    void createAccelerationIntegrationFilter()
    {   
        if (mAcceleration)
        {
            const double nyquistFrequency
                = 0.5*Preprocess::getTargetSamplingRate();
            // Highpass filter with corner at 0.75 Hz
            const std::vector<double> Wn{0.5/nyquistFrequency, 0}; 
            const int order{4};
            const double rp{20};
            const double rs{5};
            constexpr auto band = RFilterDesign::Bandtype::HIGHPASS;
            constexpr auto prototype = RFilterDesign::IIRPrototype::BUTTERWORTH;
            constexpr auto digital = RFilterDesign::IIRFilterDomain::DIGITAL;
            auto highpassFilter
                = RTSeis::FilterDesign::IIR::designSOSIIRFilter(
                     order, Wn.data(), rp, rs, band, prototype, digital); 
            // Here we'll use the elarms workflow.  Basically, wewant to do
            // 3 things:
            //   (1) Remove the instrument response with division by the gain.
            //   (2) Apply a highpass RC filter.
            //   (3) Integrate the accelerometer to velocity with the trapezoid
            //       rule.  On this point, the elarms group recommend using a
            //       smoothing parameter.  The filters are as followed by an
            //       additional smoothing.
            // More specifically: 
            //   (1) y[i] = x[i]/g where g is the gain (simple response).
            //   (2) a[i] = (y[i] - y[i-1])/b + q*a[i-1]
            //       where b = 2/(1 + q) and q ~0.994.
            //       Combining (1) and (2) we obtain
            //       a[i] = (x[i] - x[i-1])/(b*g) + q*a[i-1] 
            //   (3) v[i] = (a[i] + a[i-1]))*(dt/2)*(1/b) + q*v[i-1]
            // As filters in the Z domain we have:
            //       (1 - q*z) A = 1/(b*g)  (1 - z) X
            //       (1 - q*z) V = dt/(2*b) (1 + z) A
            //                   = 1/(1 - q*z) 1/(b*g) (1 -z )*dt/(2*b)(1 + z) X
            // So:
            //       (1 - q*z)^2 V = dt/(2*b^2*g) (1 - z)*(1 + z) X 
            // Expanding we obtain 
            //       (1 - 2q*z + q^2*z^2) V = (dt/(2*b^2 g) - dt/(2*b^2*g) z^2) X
            const double bRC  = 2/(1 + mRCFilterQ);
            const double dt   = 1/(Preprocess::getTargetSamplingRate());
            const double bCoeff = dt/(2*bRC*bRC*mSimpleResponse); 
            std::vector<double> b{bCoeff, 0, -bCoeff};
            std::vector<double> a{1, -2*mRCFilterQ, mRCFilterQ*mRCFilterQ}; 

            mHighPass = highpassFilter; 
            mIIRHighPass.initialize(mHighPass);

            mIntegrator.setNumeratorCoefficients(b);
            mIntegrator.setDenominatorCoefficients(a);
            mIIRIntegrator.initialize(mIntegrator);
        }
    } 
    /// Create the filters
    void createFilters()
    {
        if (mAcceleration){createAccelerationIntegrationFilter();}
        createVelocityFilter();
    }
    /// Process velocity signals
    void processVelocity()
    {
        int nSamples = static_cast<int> (mSignal.size());
        // Acceleration signals are already in workspace so all we need to
        // to do is copy input velocity signals
        if (!mAcceleration)
        {
            if (mWork.size() < mSignal.size()){mWork.resize(mSignal.size());}
            std::copy(mSignal.begin(), mSignal.end(), mWork.begin());
            if (mVelocitySignal.size() < mSignal.size())
            {
                mVelocitySignal.resize(mSignal.size());
            }
        }
        else
        {
            // The resizing should have been done in processAcceleration
#ifndef NDEBUG
            assert(mWork.size() == mSignal.size());
            assert(mVelocitySignal.size() == mSignal.size());
#endif
        }
        auto yWorkPtr = mVelocitySignal.data();
        // Remove mean
        mDemean.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());  
        // Window
        mWindow.apply(nSamples, mWork.data(), &yWorkPtr); 
        // If acceleration then copy
        if (mAcceleration)
        {
            std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        }
        else
        {
            double gainInverse = 1./mSimpleResponse;
            // For velocity divide by gain
            std::transform(mVelocitySignal.begin(), mVelocitySignal.end(),
                           mWork.begin(),
                           [gainInverse](double x)
                           {
                               return gainInverse*x;
                           });
        }
        // High-pass filter.  Note, the gain will be removed during
        // filtering for acceleration signals.
        mIIRVelocityFilter.apply(nSamples, mWork.data(), &yWorkPtr);
        // Get the max PGV
        mAbsoluteMaximumPGV = absoluteMaximum(mVelocitySignal);
    }
    /// Integrate acceleration signals to velocity
    void processAcceleration()
    {   
        if (!mAcceleration){return;} // Nothing to do
        int nSamples = mSignal.size();
        // Update sizes and copy input signal to workspace
        if (mWork.size() < mSignal.size()){mWork.resize(mSignal.size());}
        std::copy(mSignal.begin(), mSignal.end(), mWork.begin());
        if (mVelocitySignal.size() < mSignal.size())
        {
            mVelocitySignal.resize(mSignal.size());
        }
        auto yWorkPtr = mVelocitySignal.data();
        // Remove mean
        mDemean.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Window
        mWindow.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Highpass filter
        mIIRHighPass.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Integrate
        mIIRIntegrator.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Get absolute maximum of signal

    }
///private:
    std::vector<double> mVelocitySignal;
    std::vector<double> mWork;
    std::vector<double> mSignal;

    RTSeis::FilterRepresentations::SOS mHighPass;
    RTSeis::FilterRepresentations::BA  mIntegrator;
    RTSeis::FilterRepresentations::SOS mVelocityFilter;
    RFilters::SOSFilter<RTSeis::ProcessingMode::POST,double> mIIRHighPass;
    RFilters::IIRFilter<RTSeis::ProcessingMode::POST,double> mIIRIntegrator;
    RFilters::SOSFilter<RTSeis::ProcessingMode::POST,double> mIIRVelocityFilter;

    RFilters::Detrend<double> mDemean{RFilters::DetrendType::Constant};
    RFilters::Taper<double> mWindow;
    std::pair<double, double> mCutTimes{-1, 5};
    double mSamplingRate{100}; // Sampling rate of input signal
    double mRCFilterQ{0.994};
    double mSimpleResponse{1};
    double mAbsoluteMaximumPGV{0};
    int mTargetSignalLength{0};
    bool mAcceleration{false};
    bool mHaveVelocitySignal{false};
    bool mInitialized{false};
};

/// Constructor
Preprocess::Preprocess() :
    pImpl(std::make_unique<PreprocessImpl> ())
{
}

/// Reset class
void Preprocess::clear() noexcept
{
    pImpl = std::make_unique<PreprocessImpl> ();
}

/// Initialized
void Preprocess::initialize(const double samplingRate,
                            const double simpleResponse,
                            const std::string &unitsIn,
                            const std::pair<double, double> &cutStartEnd)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling period must be positive");
    }
    if (simpleResponse == 0)
    {
        throw std::invalid_argument("Simple response must be positive");
    }
    if (cutStartEnd.first >= cutStartEnd.second)
    {
        throw std::invalid_argument("Cut start cannot exceed cut end");
    }
    // Sort out units
    auto units = unitsIn;
    std::transform(unitsIn.begin(), unitsIn.end(), units.begin(), ::toupper);
    if (units != "DU/M/S**2" && units != "DU/M/S")
    {
        throw std::invalid_argument("units = " + unitsIn + " not handled");
    }
 
    // Passed checks - get handle on units 
    clear(); 
    pImpl->mAcceleration = true;
    if (units == "DU/M/S"){pImpl->mAcceleration = false;}
    // Some constants 
    pImpl->mSamplingRate = samplingRate;
    // The units are DU/M/S but I want DU/muM/S so divide by 1.e6
    pImpl->mSimpleResponse = simpleResponse/1.e6;
    pImpl->mCutTimes = cutStartEnd;
    // Target signal length
    pImpl->mTargetSignalLength
        = static_cast<int> (std::round( (cutStartEnd.second - cutStartEnd.first)
                                        /getTargetSamplingPeriod()) ) + 1;
    if (pImpl->mTargetSignalLength == 0)
    {
        throw std::invalid_argument("cut window is too small");
    }
    // Workspace - keep allocations to a minimum
    pImpl->mSignal.resize(pImpl->mTargetSignalLength, 0);
    pImpl->mVelocitySignal.resize(pImpl->mTargetSignalLength, 0);
    pImpl->mWork.resize(pImpl->mTargetSignalLength, 0);
    // Create filters
    pImpl->createFilters();
    // Done
    pImpl->mInitialized = true;
}

/// Initialized?
bool Preprocess::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

/// Process signal
void Preprocess::process(const std::vector<double> &signal,
                         const double arrivalTime)
{
    if (signal.empty()){throw std::invalid_argument("Signal is empty");}
    process(signal.size(), signal.data(), arrivalTime);
}

void Preprocess::process(const int n,
                         const double *__restrict__ signal,
                         const double arrivalTimeRelativeToStart)
{
    pImpl->mHaveVelocitySignal = false;
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (signal == nullptr){throw std::invalid_argument("signal is NULL");}
    if (arrivalTimeRelativeToStart < 0)
    {   
        throw std::invalid_argument("arrival time must be positive");
    }
    if (arrivalTimeRelativeToStart < -getArrivalTimeProcessingWindow().first)
    {
        throw std::invalid_argument("arrival time must be at least "
                    + std::to_string(-getArrivalTimeProcessingWindow().first));
    }
    // Easy checks
    std::vector<double> signalToCut;
    if (std::abs(getSamplingRate() - getTargetSamplingRate()) < 1.e-14)
    {
        if (n < getTargetSignalLength())
        {
            throw std::invalid_argument("Signal too small");
        }
        signalToCut.resize(n);
        std::copy(signal, signal + n, signalToCut.begin());
    }
    else
    {
        signalToCut = resample(n, signal,
                               getSamplingRate(),
                               getTargetSamplingPeriod());
        if (static_cast<int> (signalToCut.size()) < getTargetSignalLength())
        {
            throw std::invalid_argument("Interpolated signal too short");
        }
    }
    // At target sampling rate -> let's cut this signal
    auto window = getArrivalTimeProcessingWindow();
    auto tStart = arrivalTimeRelativeToStart + window.first;
    auto i1 = static_cast<int> (std::round(tStart/getTargetSamplingPeriod()));
    auto i2 = i1 + getTargetSignalLength();
    if (i1 < 0 || i2 > static_cast<int> (signalToCut.size()))
    {
        throw std::invalid_argument("Signal is too small");
    }
#ifndef NDEBUG
    assert(i2 - i1 == getTargetSignalLength());
#endif
    if (static_cast<int> (pImpl->mSignal.size()) != i2 - i1)
    {
        pImpl->mSignal.resize(i2 - i1);
    }
    std::copy(signalToCut.data() + i1, signalToCut.data() + i2,
              pImpl->mSignal.begin());
    // Check for dead signal
    bool lDead = true;
    const double *__restrict__ signalPtr = pImpl->mSignal.data();
    for (int i = 1; i < getTargetSignalLength(); ++i)
    {
        if (signalPtr[i] != signalPtr[i - 1]){lDead = false;}
    }
    if (lDead){throw std::invalid_argument("Signal is dead");}
    // Process this signal
    if (pImpl->mAcceleration){pImpl->processAcceleration();} // Integrate
    pImpl->processVelocity(); // Process velocity signal 
    pImpl->mHaveVelocitySignal = true;
}

/// Get the velocity signal
std::vector<double> Preprocess::getVelocitySignal() const
{
    std::vector<double> result;
    getVelocitySignal(&result);
    return result;
}

void Preprocess::getVelocitySignal(std::vector<double> *velocitySignal) const
{
    if (!haveVelocitySignal())
    {   
        throw std::runtime_error("Velocity signal not computed");
    }
    if (velocitySignal->size() != pImpl->mVelocitySignal.size())
    {
        velocitySignal->resize(pImpl->mVelocitySignal.size());
    }
    std::copy(pImpl->mVelocitySignal.begin(),
              pImpl->mVelocitySignal.end(),
              velocitySignal->begin());
}

/// Get max PGV
double Preprocess::getAbsoluteMaximumPeakGroundVelocity() const
{
    if (!haveVelocitySignal())
    {
        throw std::runtime_error("Velocity signal not computed");
    }
    return pImpl->mAbsoluteMaximumPGV;
}

/// Have velocity signal?
bool Preprocess::haveVelocitySignal() const noexcept
{
    return pImpl->mHaveVelocitySignal;
}

/// Sampling rate of input signal
double Preprocess::getSamplingRate() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mSamplingRate;
}

std::pair<double, double>
Preprocess::getArrivalTimeProcessingWindow() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mCutTimes;
}

/// Target signal length
int Preprocess::getTargetSignalLength() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mTargetSignalLength;
}


/// Destructor
Preprocess::~Preprocess() = default;

double Preprocess::getTargetSamplingRate() noexcept
{
    return TARGET_SAMPLING_RATE;
}

double Preprocess::getTargetSamplingPeriod() noexcept
{
    return TARGET_SAMPLING_PERIOD;
}
