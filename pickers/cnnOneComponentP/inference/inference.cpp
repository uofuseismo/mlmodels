#include "uussmlmodels/pickers/cnnOneComponentP/inference.hpp"
#include "private/h5io.hpp"
#define EXPECTED_SIGNAL_LENGTH 400
#define SAMPLING_RATE 100
#define N_CHANNELS 1
#define MIN_PERTURBATION -0.75
#define MAX_PERTURBATION  0.75
using namespace UUSSMLModels::Pickers::CNNOneComponentP;
#include "openvino.hpp"

class Inference::InferenceImpl
{
public:
    /// Constructor
    explicit InferenceImpl(const Inference::Device device) :
        mOpenVINO(device, MIN_PERTURBATION, MAX_PERTURBATION)
    {
    }
    OpenVINOImpl mOpenVINO;
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
U Inference::predict(const std::vector<U> &vertical) const
{
    if (!isInitialized()){throw std::runtime_error("Class not initialized");}
    return pImpl->mOpenVINO.predict(vertical);
}

///--------------------------------------------------------------------------///
///                           Template Instantiation                         ///
///--------------------------------------------------------------------------///
template double UUSSMLModels::Pickers::CNNOneComponentP::Inference::predict(
    const std::vector<double> &) const;
template float UUSSMLModels::Pickers::CNNOneComponentP::Inference::predict(
    const std::vector<float> &) const;
