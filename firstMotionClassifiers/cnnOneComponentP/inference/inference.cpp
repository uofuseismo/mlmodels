#include <cmath>
#ifndef NDEBUG
#include <cassert>
#endif
#include "uussmlmodels/firstMotionClassifiers/cnnOneComponentP/inference.hpp"
#include "private/h5io.hpp"
#define EXPECTED_SIGNAL_LENGTH 400
#define SAMPLING_RATE 100
#define N_CHANNELS 1
using namespace UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP;
#include "openvino.hpp"

class Inference::InferenceImpl
{
public:
    /// Constructor
    explicit InferenceImpl(const Inference::Device device) :
        mOpenVINO(device),
        mDevice(device)
    {
    }
    OpenVINOImpl mOpenVINO;
    double mProbabilityThreshold{1./3.};
    const Inference::Device mDevice{Inference::Device::CPU};
    bool mUseOpenVINO{false};
    bool mInitialized{false};
};

/// Constructor
Inference::Inference() :
    Inference(Inference::Device::CPU)
{
}

/// Constructor with given device
Inference::Inference(const Inference::Device device) :
    pImpl(std::make_unique<InferenceImpl> (device))
{
}

/// Destructor
Inference::~Inference() = default;

/// Reset class
void Inference::clear() noexcept
{
    auto device = pImpl->mDevice;
    pImpl = std::make_unique<InferenceImpl> (device);
}

/// Expected signal length
int Inference::getExpectedSignalLength() noexcept
{
    return EXPECTED_SIGNAL_LENGTH;
}

/// Sampling rate
double Inference::getSamplingRate() noexcept
{
    return SAMPLING_RATE;
}

/// Load model
void Inference::load(const std::string &fileName,
                     const Inference::ModelFormat format)
{
    if (!std::filesystem::exists(fileName))
    {
        throw std::runtime_error("Model file: " + fileName + " does not exist");
    }
    if (format == Inference::ModelFormat::ONNX)
    {
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.load(fileName);
        pImpl->mUseOpenVINO = true;
        pImpl->mInitialized = true;
#else
        throw std::runtime_error("Recompile with OpenVino to read ONNX");
#endif
    }
    else if (format == Inference::ModelFormat::HDF5)
    {
        Weights weights;
        weights.loadFromHDF5(fileName);
#ifdef WITH_OPENVINO
        pImpl->mOpenVINO.createFromWeights(weights, 1);
        pImpl->mUseOpenVINO = true;
        pImpl->mInitialized = true;
#else
        throw std::runtime_error("Recompile with OpenVino to load HDF5");
#endif
    }
    else
    {
        throw std::runtime_error("Unhandled model format");
    }
}

/// Initialized?
bool Inference::isInitialized() const noexcept
{
    return pImpl->mInitialized;
}

/// Perform inference
template<typename U>
std::tuple<U, U, U> Inference::predictProbability(
    const std::vector<U> &vertical) const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mOpenVINO.predictProbability(vertical);
}

/// Probability threshold
void Inference::setProbabilityThreshold(const double threshold)
{
    if (threshold < 0 || threshold > 1)
    {
        throw std::invalid_argument("Threshold must be in range [0,1]");
    }
    pImpl->mProbabilityThreshold = threshold;
}

double Inference::getProbabilityThreshold() const noexcept
{
    return pImpl->mProbabilityThreshold;
}

/// Predicts up/down/unkonwn
template<typename U>
Inference::FirstMotion 
Inference::predict(const std::vector<U> &vertical) const
{
    auto [pUp, pDown, pUnknown] = predictProbability(vertical);
    auto threshold = getProbabilityThreshold();
    return convertProbabilityToClass(pUp, pDown, pUnknown, threshold);
/*
    auto pUp8 = static_cast<double> (pUp);
    auto pDown8 = static_cast<double> (pDown);
    auto pUnknown8 = static_cast<double> (pUnknown);
    auto threshold = getProbabilityThreshold();
    if (pUp8 > pDown8)
    {
        if (pUp8 > std::max(pUnknown8, threshold))
        {
            return FirstMotion::Up;
        }
        return FirstMotion::Unknown;
    }
    else
    {
        if (pDown8 > std::max(pUnknown8, threshold))
        {
            return FirstMotion::Down;
        }
        return FirstMotion::Unknown;
    }
#ifndef NDEBUG
    assert(false);
#endif
*/
}

template<typename T>
Inference::FirstMotion 
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::convertProbabilityToClass(
    const T probabilityUp,
    const T probabilityDown,
    const T probabilityUnknown,
    const double threshold)
{
    constexpr T eps = std::numeric_limits<T>::epsilon()*100;
    if (threshold < 0 || threshold > 1)
    {
        throw std::invalid_argument("Threshold must be in range [0,1]");
    }
    if (probabilityUp < 0 || probabilityUp > 1)
    {
        throw std::invalid_argument("Up probability must be in range [0,1]");
    }
    if (probabilityDown < 0 || probabilityDown > 1)
    {
        throw std::invalid_argument("Down probability must be in range [0,1]");
    }
    if (probabilityUnknown < 0 || probabilityUnknown > 1)
    {
        throw std::invalid_argument(
            "Unknown probability must be in range [0,1]");
    }
    auto pSum = probabilityUp + probabilityDown + probabilityUnknown;
    if (std::abs(pSum - 1) > eps)
    {
        throw std::invalid_argument("Probabilities do not sum to unity ("
                                  + std::to_string(probabilityUp) + ","
                                  + std::to_string(probabilityDown) + ","
                                  + std::to_string(probabilityUnknown) + ")");
    }

    auto pUp8 = static_cast<double> (probabilityUp);
    auto pDown8 = static_cast<double> (probabilityDown);
    auto pUnknown8 = static_cast<double> (probabilityUnknown);
    if (pUp8 > pDown8)
    {
        if (pUp8 > std::max(pUnknown8, threshold))
        {
            return Inference::FirstMotion::Up;
        }
        return Inference::FirstMotion::Unknown;
    }
    else
    {
        if (pDown8 > std::max(pUnknown8, threshold))
        {
            return Inference::FirstMotion::Down;
        }
        return Inference::FirstMotion::Unknown;
    }
#ifndef NDEBUG
    assert(false);
#endif
}

///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template std::tuple<double, double, double>
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::predictProbability(
    const std::vector<double> &) const;
template std::tuple<float, float, float>
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::predictProbability(
    const std::vector<float> &) const;

template Inference::FirstMotion
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::predict(
    const std::vector<double> &) const;
template Inference::FirstMotion
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::predict(
    const std::vector<float> &) const;

template
Inference::FirstMotion 
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::convertProbabilityToClass(
    double probabilityUp,
    double probabilityDown,
    double probabilityUnknown,
    double threshold);
template
Inference::FirstMotion 
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::convertProbabilityToClass(
    float probabilityUp,
    float probabilityDown,
    float probabilityUnknown,
    double threshold);


