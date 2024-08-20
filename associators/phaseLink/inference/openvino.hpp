#ifdef WITH_OPENVINO
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <openvino/openvino.hpp>
namespace
{
/// @brief Defines the OpenVINO model implementation.
class OpenVINOImpl
{
public:
    explicit OpenVINOImpl(
        const Region region,
        const Inference::Device requestedDevice = Inference::Device::CPU)
    {
        // Preallocate input space based on the region
        if (region == Region::Utah)
        {
            mSimulationSize = UTAH_SIMULATION_SIZE;
            mInputShape
                = ov::Shape{1, UTAH_SIMULATION_SIZE, N_FEATURES};
        }
        else
        {
            mSimulationSize = YELLOWSTONE_SIMULATION_SIZE;
            mInputShape
                = ov::Shape{1, YELLOWSTONE_SIMULATION_SIZE, N_FEATURES};
        }
        mInputTensor = ov::Tensor{mInputType, mInputShape};
        // Find the device
        auto availableDevices = mCore.get_available_devices();
        bool matchedDevice{false};
        bool haveCPU{false};
        mDevice = "CPU";
        for (const auto &device : availableDevices)
        {
            if (device == "CPU")
            {
                haveCPU = true;
                if (requestedDevice == Inference::Device::CPU)
                {
                    mDevice = device;
                    matchedDevice = true;
                } 
            }
            else if (device == "GPU")
            {
                if (requestedDevice == Inference::Device::GPU)
                {
                    mDevice = device;
                    matchedDevice = true;
                }
            }
        }
        if (!matchedDevice)
        {
            if (!haveCPU)
            {
                throw std::runtime_error("Desired device not supported");
            }
            else
            {
                std::cerr << "Default to CPU" << std::endl;
            }
        }
    }
    /// @brief Loads the ONNX file.
    void load(const std::string &fileName)
    {
        if (!std::filesystem::exists(fileName))
        {
            throw std::runtime_error("Model " + fileName + " file not set");
        }
        // Load the model and put it on the device
        auto model = mCore.read_model(fileName);
        mFileName = fileName;
        mCompiledModel = mCore.compile_model(model, mDevice);
    }
    /// @brief Perform inference
    template<typename T>
    std::vector<T> predictProbability(const std::vector<T> &X)
    {
        constexpr T zero{0}; 
        if (mSimulationSize*N_FEATURES != static_cast<int> (X.size()))
        {
            throw std::invalid_argument("Unknown input shape");
        }
        std::vector<T> result(mSimulationSize, zero);
        mInferenceRequest = mCompiledModel.create_infer_request();
        auto data = reinterpret_cast<float *> (mInputTensor.data());
        std::copy(X.begin(), X.end(), data);
    
        mInferenceRequest.set_input_tensor(mInputTensor);
        mInferenceRequest.infer();
        const auto &inferenceResult = mInferenceRequest.get_output_tensor();
        auto inferenceResultPtr
            = reinterpret_cast<const float *> (inferenceResult.data());
        std::copy(inferenceResultPtr, inferenceResultPtr + mSimulationSize,
                  result.begin());
        return result;
    }
    template<typename T>
    std::vector<T> predictProbability(const int nRows,
                                      const std::vector<T> &Xin)
    {
        if (nRows == mSimulationSize)
        {
            return predictProbability(Xin);
        }
        else
        {
            std::vector<float> X(mSimulationSize*N_FEATURES, 0.0f);
            auto nCopy = N_FEATURES*std::min(nRows, mSimulationSize);
            std::copy(Xin.data(), Xin.data() + nCopy, X.data());
            // Null out rest
            for (int i = nRows; i < mSimulationSize; ++i)
            {
                X[N_FEATURES*i + 4] = 1;
            }
            auto probability = predictProbability<float> (X);
            std::vector<T> result(probability.size());
            std::copy(probability.begin(), probability.end(), result.begin());
            return result;
        }
    }
    /*
    template<typename T>
    std::vector<int> predict(const size_t nRows,
                             const std::vector<T> &X,
                             const double thresholdIn = 0.5)
    {
        if (thresholdIn <= 0 || thresholdIn >= 1)
        {
            throw std::invalid_argument("Threshold must be in range (0, 1)");
        }
        auto probabilities = predictProbability(nRows, X);
        const T threshold{static_cast<T> (thresholdIn)};
        std::vector<int> result(probabilities.size());
        for (int i = 0; i < static_cast<int> (probabilities.size()); ++i)
        {
            result[i] = 0;
            if (probabilities[i] > threshold){result[i] = 1;}
        }
        return result;
    }
    template<typename T>
    std::vector<int> predict(const std::vector<T> &X,
                             const double threshold = 0.5)
    {
        return predict(mSimulationSize, X, threshold);
    }
    */
///private:
    const ov::element::Type mInputType{ov::element::f32};
    ov::Shape mInputShape;
    ov::Tensor mInputTensor;
    ov::Core mCore;
    ov::CompiledModel mCompiledModel;
    ov::InferRequest mInferenceRequest;
    std::filesystem::path mFileName;
    std::string mDevice{"CPU"};
    int mSimulationSize{0};
};
}
#else
namespace 
{
class OpenVINOImpl
{
public:
};
}
#endif
