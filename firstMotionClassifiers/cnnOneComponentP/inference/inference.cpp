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
        mOpenVINO(device)
    {
    }
    OpenVINOImpl mOpenVINO;
    double mProbabilityThreshold{1./3.};
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
        throw std::runtime_error("Recompile with OpenVino to read ONNX");
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

///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template std::tuple<double, double, double>
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::predictProbability(
    const std::vector<double> &) const;
template std::tuple<float, float, float>
UUSSMLModels::FirstMotionClassifiers::CNNOneComponentP::Inference::predictProbability(
    const std::vector<float> &) const;
