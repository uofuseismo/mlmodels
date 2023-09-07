#include <iostream>
#include <cmath>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#endif
#include <rtseis/postProcessing/singleChannel/waveform.hpp>
#include <rtseis/transforms/slidingWindowRealDFTParameters.hpp>
#include <rtseis/transforms/spectrogram.hpp>
#include "uussmlmodels/eventClassifiers/cnnThreeComponent/preprocessing.hpp"

#define TARGET_SAMPLING_RATE 100
#define TARGET_SAMPLING_PERIOD 0.01

using namespace UUSSMLModels::EventClassifiers::CNNThreeComponent;

class Preprocessing::PreprocessingImpl
{
public:

///private:
    /// Holds the data processing waveform
    RTSeis::PostProcessing::SingleChannel::Waveform<double> mWave;
    /// Anticipating 100 Hz sampling rate
    double mTargetSamplingPeriod{TARGET_SAMPLING_PERIOD};
    /// Remove trend
    bool mRemoveTrend{true};
    /// Taper the data prior to filtering
    double mTaperPct{1};
    bool mDoTaper{true};
    /// Bandpass filter data
    int mFilterPoles{2};
    /// Number of cascades
    int mCascades{1};
    std::pair<double, double> mCorners = {1, 20}; // 1 Hz to 20 Hz
    RTSeis::PostProcessing::SingleChannel::IIRPrototype mPrototype
    {   
        RTSeis::PostProcessing::SingleChannel::IIRPrototype::BUTTERWORTH
    };
    double mRipple{5}; // Ripple for Chebyshev 1 or 2 - not applicable
    const bool mZeroPhase{true}; // Filter does not need to be applied causally
    // This is not for real-time
    RTSeis::PostProcessing::SingleChannel::InterpolationMethod
       mInterpolationMethod
    {   
        RTSeis::PostProcessing::SingleChannel::InterpolationMethod::WEIGHTED_AVERAGE_SLOPES
    }; 
};

