#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#endif
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
#include "uuss/features/magnitude/verticalChannelFeatures.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"
#include "private/magnitudeUtilities.hpp"

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100
#define TARGET_SIGNAL_LENGTH 500    // 1s before to 4s after
#define PRE_ARRIVAL_TIME 1          // 1s before P arrival
#define POST_ARRIVAL_TIME 4         // 4s after P arrival
#define P_PICK_ERROR 0.05           // Alysha's P pickers are usually within 5 samples
#define TARGET_SIGNAL_DURATION (TARGET_SIGNAL_LENGTH - 1)*TARGET_SAMPLING_PERIOD

using namespace UUSS::Features::Magnitude;

namespace
{
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

}

class VerticalChannelFeatures::FeaturesImpl
{
public:
    FeaturesImpl()
    {
        mDemean.initialize(
            RTSeis::FilterImplementations::DetrendType::CONSTANT);
        const double pct = 5;
        mHanning.initialize(pct,
                        RTSeis::FilterImplementations::TaperWindowType::Hann);
        mSignal.resize(mTargetSignalLength, 0);
        mVelocitySignal.resize(mTargetSignalLength, 0);
        mWork.resize(mTargetSignalLength, 0);
        // Setup CWT
        const double omega0 = 4; // tradeoff between time and frequency
        mMorlet.setParameter(omega0);
        mMorlet.enableNormalization();
        mMorlet.disableNormalization();
        std::vector<double> scales(mCenterFrequencies.size());
        for (int i = 0; i < static_cast<int> (scales.size()); ++i)
        {
            scales[i] = (omega0*mSamplingRate)/(2*M_PI*mCenterFrequencies[i]);
        }
        mCWT.initialize(mSignal.size(),
                        scales.size(), scales.data(),
                        mMorlet,
                        mTargetSamplingRate);
        mAmplitudeCWT.resize(scales.size()*mSignal.size(), 0);
        mCumulativeAmplitudeCWT.resize(mAmplitudeCWT.size(), 0);
    }
    void processAcceleration()
    {
        if (!mAcceleration){return;} // Nothing to do
        int nSamples = mSignal.size();
        mWork = mSignal;
        auto yWorkPtr = mVelocitySignal.data();
        // Remove mean
        mDemean.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Window
        mHanning.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Highpass filter
        mIIRHighPass.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Integrate
        mIIRIntegrator.apply(nSamples, mWork.data(), &yWorkPtr);
    }
    void processVelocity()
    {
        int nSamples = mSignal.size();
        if (mAcceleration)
        {
            mWork = mVelocitySignal;
        }
        else
        {
            mWork = mSignal;
        }
        auto yWorkPtr = mVelocitySignal.data();
        // Remove mean
        mDemean.apply(nSamples, mWork.data(), &yWorkPtr);
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());  
        // Window
        mHanning.apply(nSamples, mWork.data(), &yWorkPtr); 
        std::copy(yWorkPtr, yWorkPtr + nSamples, mWork.begin());
        // Remove gain
        if (!mAcceleration)
        {
            auto gainInverse = 1./mSimpleResponse;
            std::transform(mWork.begin(), mWork.end(), mWork.begin(),
                           [gainInverse](double x)
                           {
                               return gainInverse*x;
                           });
        }
        // High-pass filter
        mIIRVelocityFilter.apply(nSamples, mWork.data(), &yWorkPtr);
    }
    // Compute the scalogram
    void computeVelocityScalogram()
    {
        const double dt2 = 1./(2*mSamplingRate);
        // Calculate the scalogram
        mCWT.transform(mVelocitySignal.size(), mVelocitySignal.data());
        auto nScales  = mCWT.getNumberOfScales(); 
        auto nSamples = mCWT.getNumberOfSamples();
#ifndef NDEBUG
        assert(mAmplitudeCWT.size() == nScales*nSamples);
        assert(mCumulativeAmplitudeCWT.size() == nScales*nSamples);
#endif
        auto aPtr = mAmplitudeCWT.data();
        mCWT.getAmplitudeTransform(nSamples, nScales, &aPtr);
        // Accmulate in frequency bins as a function of time
        std::fill(mCumulativeAmplitudeCWT.begin(),
                  mCumulativeAmplitudeCWT.end(), 0);
        for (int j = 0; j < nScales; ++j)
        {
            int indx = j*nSamples;
            const double *aScale = mAmplitudeCWT.data() + indx;
            double *cumPtr = mCumulativeAmplitudeCWT.data() + indx;
            int iStart = 0;
            cumPtr[iStart] = aScale[iStart];
            for (int i = iStart + 1; i < nSamples; ++i)
            {
                cumPtr[i] = cumPtr[i-1] + (aScale[i] + aScale[i-1])*dt2;
            }
        }
        // Dump the scalogram for debugging
        std::string fname{"ampVel.txt"};
        if (mAcceleration){fname = "ampAcc.txt";}
        std::ofstream ofl;
        ofl.open(fname);
        for (int i = 0; i < nSamples; ++i)
        {
            for (int j = 0; j < nScales; ++j)
            {
                ofl << i/mSamplingRate << " " 
                    << mCenterFrequencies[j] << " "
                    << aPtr[j*nSamples + i] << " "
                    << mCumulativeAmplitudeCWT[j*nSamples + i] << std::endl;
            }
            ofl << std::endl;
        }
        ofl.close();
    }
    // Simple function to do pre-processing then extract features
    void process()
    {
        if (mAcceleration){processAcceleration();}
        processVelocity();
        computeVelocityScalogram();
        extractFeatures();
    }
    void extractFeatures()
    {
#ifndef NDEBUG
        assert(mVelocitySignal.size() == TARGET_SIGNAL_LENGTH);
#endif
        auto nSamples = TARGET_SIGNAL_LENGTH;
        auto iPick
            = static_cast<int >(std::round((PRE_ARRIVAL_TIME - P_PICK_ERROR)
                                           /TARGET_SAMPLING_PERIOD));
        // Get the signal variance before after the pick.  The variance is
        // the signal power - the DC power.
        auto varianceNoise  = variance(iPick, mVelocitySignal.data());
        auto varianceSignal = variance(nSamples - iPick,
                                       mVelocitySignal.data() + iPick);
        // Get difference of min/max amplitude 
        const auto [vMinNoise,  vMaxNoise]
            = std::minmax_element(mVelocitySignal.begin(),
                                  mVelocitySignal.begin() + iPick);
        // Get the nominal period and amplitude of arrival
        std::cout << "duration,minNoise,maxNoise,dMinMaxNoise,minSignal,maxSignal,dMinMaxSignal,varianceNoise,varianceSignal,dominantPeriod,amplitudeAtDominantPeriod,dominantCumulativeEnergyPeriod,cumulativeEnergyAtDominantPeriod" << std::endl;
        auto nScales  = mCWT.getNumberOfScales();
        for (const auto &duration : mDurations)
        {
            auto iStart = iPick;
            auto iEnd = static_cast<int>
                        (std::round( (PRE_ARRIVAL_TIME + duration)
                                     /TARGET_SAMPLING_PERIOD));
            iEnd = std::min(iEnd, nSamples);
            auto nSubSamples = iEnd - iStart;
            // Get min/max signal
            const auto [vMinSignal, vMaxSignal]
                = std::minmax_element(mVelocitySignal.data() + iStart,
                                      mVelocitySignal.data() + iEnd);
            // Get variance in signal
            auto varianceSignal = variance(nSubSamples,
                                           mVelocitySignal.data() + iStart);
            std::pair<double, double> dominantPeriodAmplitude{0, 0};
            for (int j = 0; j < nScales; ++j)
            {
                for (int i = iStart; i < iEnd; ++i)
                {
                    auto indx = j*nSamples + i;
                    if (mAmplitudeCWT[indx] > dominantPeriodAmplitude.second)
                    {
                        dominantPeriodAmplitude.first  = 1./mCenterFrequencies[j];
                        dominantPeriodAmplitude.second = mAmplitudeCWT[indx];
                    }
                }
            }
            // Look at cumulative energy in bands
            std::vector<std::pair<double, double>> cumulativeEnergy(nScales);
            for (int j = 0; j < nScales; ++j)
            {
                 auto indx = j*nSamples + iEnd;   // Time of interest
                 auto jndx = j*nSamples + iStart; // Subtract from cumulative
                 auto dEnergy = mCumulativeAmplitudeCWT[indx]
                              - mCumulativeAmplitudeCWT[jndx];
                 cumulativeEnergy[j].first  = 1./mCenterFrequencies[j];
                 cumulativeEnergy[j].second = dEnergy;
            }
            std::sort(cumulativeEnergy.begin(), cumulativeEnergy.end(), 
                      [](const std::pair<double, double> &a,
                         std::pair<double, double> &b)
                      {
                          return a.second > b.second;
                      });
            std::cout << duration << " "
                      << *vMinNoise  << " " << *vMaxNoise << " " << *vMaxNoise - *vMinNoise << " "
                      << *vMinSignal << " " << *vMaxSignal << " " << *vMaxSignal - *vMinSignal << " " 
                      << varianceNoise << " " << varianceSignal << " " 
                      << dominantPeriodAmplitude.first << " " << dominantPeriodAmplitude.second << " "
                      << cumulativeEnergy[0].first << " " << cumulativeEnergy[0].second << " "
                      << cumulativeEnergy[1].first << " " << cumulativeEnergy[1].second << " "
                      << cumulativeEnergy[2].first << " " << cumulativeEnergy[2].second << " "
                      <<std::endl;
 
        }
/*
std::cout << "work it:" << std::endl;
std::cout << varianceNoise << " " << varianceSignal << std::endl;
std::cout << *vMinNoise << " " << *vMaxNoise << " " << *vMaxNoise - *vMinNoise << std::endl;
std::cout << *vMinSignal << " " << *vMaxSignal << " " << *vMaxSignal - *vMinSignal << std::endl;
*/
    }
    void createVelocityFilter()
    {
        double nyquistFrequency = 0.5*mTargetSamplingRate;
        auto normalizedLowCorner  = 1/nyquistFrequency;
        auto normalizedHighCorner = std::min(nyquistFrequency*0.9,
                                             18/nyquistFrequency);
        const std::vector<double> Wn{normalizedLowCorner,
                                     normalizedHighCorner};
        const int order{3}; // Will be doubled in design
        const double rp{20};
        const double rs{5};
        const RTSeis::FilterDesign::Bandtype band
            = RTSeis::FilterDesign::Bandtype::BANDPASS;
        const RTSeis::FilterDesign::IIRPrototype prototype
            = RTSeis::FilterDesign::IIRPrototype::BUTTERWORTH;
        const RTSeis::FilterDesign::IIRFilterDomain digital
            = RTSeis::FilterDesign::IIRFilterDomain::DIGITAL;
        mVelocityFilter = RTSeis::FilterDesign::IIR::designSOSIIRFilter(
                 order, Wn.data(), rp, rs, band, prototype, digital);
        // Initialize velocity filter
        mIIRVelocityFilter.initialize(mVelocityFilter);
    }
    void createAccelerationIntegrationFilter()
    {
        if (mAcceleration)
        {
            double nyquistFrequency = 0.5*mTargetSamplingRate;
            // Highpass filter with corner at 0.75 Hz
            const std::vector<double> Wn{0.5/nyquistFrequency, 0};
            const int order{4};
            const double rp{20};
            const double rs{5};
            const RTSeis::FilterDesign::Bandtype band
                = RTSeis::FilterDesign::Bandtype::HIGHPASS;
            const RTSeis::FilterDesign::IIRPrototype prototype
                = RTSeis::FilterDesign::IIRPrototype::BUTTERWORTH;
            const RTSeis::FilterDesign::IIRFilterDomain digital
                = RTSeis::FilterDesign::IIRFilterDomain::DIGITAL;
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
            const double dt   = 1/(mTargetSamplingRate);
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
    void update()
    {
        createAccelerationIntegrationFilter();
        createVelocityFilter();
    } 
//private:
    // Center frequencies in CWT
    const std::vector<double> mCenterFrequencies{1, 1.25, 1.5, 1.75,
                                                 2, 2.25, 2.5, 2.75,
                                                 3, 3.25, 3.5, 3.75,
                                                 4, 4.25, 4.5, 4.75,
                                                 5, 5.25, 5.5, 5.75,
                                                 6, 6.25, 6.5, 6.75,
                                                 7, 7.25, 7.5, 7.75,
                                                 8, 8.25, 8.5, 8.75,
                                                 9, 9.25, 9.5, 9.75,
                                                 10, 10.5,
                                                 11, 11.5,
                                                 12, 12.5,
                                                 13, 13.5,
                                                 14, 14.5,
                                                 15, 16, 17, 18};
    // Initially signal is copied to mSignal but mSignal ends up as workspace
    std::vector<double> mSignal;
    // Holds signals with units of velocity
    std::vector<double> mVelocitySignal;
    // Workspace
    std::vector<double> mWork;
    std::vector<double> mAmplitudeCWT;
    std::vector<double> mCumulativeAmplitudeCWT;
    std::array<double, 3> mDurations{1, 2, 3};
    std::string mUnits;
    RTSeis::FilterRepresentations::SOS mHighPass;
    RTSeis::FilterRepresentations::BA  mIntegrator;
    RTSeis::FilterRepresentations::SOS mVelocityFilter;
    RTSeis::FilterImplementations::SOSFilter
        <RTSeis::ProcessingMode::POST, double> mIIRHighPass;
    RTSeis::FilterImplementations::IIRFilter
        <RTSeis::ProcessingMode::POST, double> mIIRIntegrator;
    RTSeis::FilterImplementations::SOSFilter
        <RTSeis::ProcessingMode::POST, double> mIIRVelocityFilter;
    RTSeis::FilterImplementations::Detrend<double> mDemean;
    RTSeis::FilterImplementations::Taper<double> mHanning;
    RTSeis::Transforms::Wavelets::Morlet mMorlet;
    RTSeis::Transforms::ContinuousWavelet<double> mCWT;
    double mRCFilterQ{0.994};
    double mSamplingRate{100};
    double mSimpleResponse{1}; // Proportional to micrometers
    const double mTargetSamplingRate{TARGET_SAMPLING_RATE};
    const double mTargetSamplingPeriod{TARGET_SAMPLING_PERIOD};
    const double mTargetSignalDuration{TARGET_SIGNAL_DURATION};
    const int mTargetSignalLength{TARGET_SIGNAL_LENGTH};
    bool mAcceleration{false};
    bool mHaveSignal{false};
    bool mInitialized{false};
};

/// C'tor
VerticalChannelFeatures::VerticalChannelFeatures() :
    pImpl(std::make_unique<FeaturesImpl> ())
{
}

/// Reset class
void VerticalChannelFeatures::clear() noexcept
{
    pImpl = std::make_unique<FeaturesImpl> ();
}

/// Destructor
VerticalChannelFeatures::~VerticalChannelFeatures() = default;

/// Set the signal
template<typename U>
void VerticalChannelFeatures::process(const std::vector<U> &signal,
                                      const double arrivalTimeRelativeToStart)
{
    if (signal.empty()){throw std::runtime_error("Signal is empty");}
    process(signal.size(), signal.data(), arrivalTimeRelativeToStart);
}

template<typename U>
void VerticalChannelFeatures::process(
    const int n, const U *__restrict__ signal,
    const double arrivalTimeRelativeToStart)
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    if (arrivalTimeRelativeToStart < 0)
    {
        throw std::invalid_argument("arrival time must be positive");
    }
    if (arrivalTimeRelativeToStart < -getArrivalTimeProcessingWindow().first)
    {
        throw std::invalid_argument("arrival time must be at least "
                    + std::to_string(-getArrivalTimeProcessingWindow().first));
    }
    if (signal == nullptr){throw std::invalid_argument("signal is NULL");}
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
    assert(i2 - i1 == TARGET_SIGNAL_LENGTH);
#endif
    if (static_cast<int> (pImpl->mSignal.size()) != i2 - i1)
    {
        pImpl->mSignal.resize(i2 - i1); 
    }
    std::copy(signalToCut.data() + i1, signalToCut.data() + i2,
              pImpl->mSignal.begin());
    // Process the data
    pImpl->mHaveSignal = false;
    pImpl->process();
    pImpl->mHaveSignal = true; 
}

/// Have signal?
bool VerticalChannelFeatures::haveSignal() const noexcept
{
    return pImpl->mHaveSignal;
}

/// Velocity signal
std::vector<double> VerticalChannelFeatures::getVelocitySignal() const
{
    return pImpl->mVelocitySignal;
}

/// Sampling rate
void VerticalChannelFeatures::initialize(const double samplingRate,
                                         const double simpleResponse,
                                         const std::string &unitsIn)
{
    if (samplingRate <= 0)
    {
        throw std::invalid_argument("Sampling rate must be positive");
    }
    if (simpleResponse == 0)
    {
        throw std::invalid_argument("Simple response cannot be zero");
    }
    std::string units{unitsIn};
    std::transform(unitsIn.begin(), unitsIn.end(), units.begin(), ::toupper);
    bool isAcceleration{false};
    if (units == "DU/M/S**2")
    {
        isAcceleration = true;
    }
    else if (units == "DU/M/S")
    {
        isAcceleration = false;
    }
    else
    {
        throw std::runtime_error("units = " + unitsIn + " not handled");
    }
    clear();
    pImpl->mUnits = unitsIn;
    // Make response proportional to micrometers.  Response units are 
    // 1/meter so to go to 1/millimeter we do 1/(meter*1e6) which effectively
    // divides the input by 1e6.
    pImpl->mSimpleResponse = simpleResponse/1e6; // proportional to micrometers
    pImpl->mAcceleration = isAcceleration;
    pImpl->mSamplingRate = samplingRate;
    pImpl->update();
    pImpl->mInitialized = true;
}

bool VerticalChannelFeatures::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

double VerticalChannelFeatures::getSamplingRate() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mSamplingRate;
}

double VerticalChannelFeatures::getSimpleResponse() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mSimpleResponse*1e6; // micrometers to meters
}

std::string VerticalChannelFeatures::getSimpleResponseUnits() const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mUnits;
}

/// Target information
double VerticalChannelFeatures::getTargetSamplingRate() noexcept
{
    return TARGET_SAMPLING_RATE;
}

double VerticalChannelFeatures::getTargetSamplingPeriod() noexcept
{
    return TARGET_SAMPLING_PERIOD;
}

double VerticalChannelFeatures::getTargetSignalDuration() noexcept
{
    return TARGET_SIGNAL_DURATION;
}

int VerticalChannelFeatures::getTargetSignalLength() noexcept
{
    return TARGET_SIGNAL_LENGTH;
}

std::pair<double, double>
VerticalChannelFeatures::getArrivalTimeProcessingWindow() noexcept
{
    return std::pair<double, double> {-PRE_ARRIVAL_TIME, POST_ARRIVAL_TIME};
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template void UUSS::Features::Magnitude::VerticalChannelFeatures::process(
    const std::vector<double> &, double); 
template void UUSS::Features::Magnitude::VerticalChannelFeatures::process(
    const std::vector<float> &, double);
