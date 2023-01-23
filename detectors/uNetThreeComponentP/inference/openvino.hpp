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
    OpenVINOImpl(const Inference::Device requestedDevice = Inference::Device::CPU)
    {
        auto availableDevices = mCore.get_available_devices();
        bool matchedDevice = false;
        bool haveCPU = false;
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
/*
    /// @brief Sets the ONNX file name.
    void setModelFile(const std::string &fileName)
    {
        if (!std::filesystem::exists(fileName))
        {
            throw std::invalid_argument("Model " + fileName
                                      + " does not exist");
        }
        mFileName = fileName;
    }
*/
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
        // Get an inference request
        mInferenceRequest = mCompiledModel.create_infer_request();
    }
    /// @brief Sets the data
    template<typename T>
    void setSignals(const std::vector<T> &vertical,
                    const std::vector<T> &north,
                    const std::vector<T> &east)
    {
        if (vertical.size() != MINIMUM_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("Vertical is wrong size");
        }
        if (north.size() != MINIMUM_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("North is wrong size");
        }
        if (east.size() != MINIMUM_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("East is wrong size");
        }
        auto data = reinterpret_cast<float *> (mInputTensor.data());
        rescaleAndCopy(MINIMUM_SIGNAL_LENGTH,
                       vertical.data(), north.data(), east.data(), data);
    }
    /// @brief Perform inference
    template<typename T>
    void predictProbability(const std::vector<T> &vertical,
                            const std::vector<T> &north,
                            const std::vector<T> &east,
                            std::vector<float> *probability)
    {
        mInferenceRequest = mCompiledModel.create_infer_request();
        setSignals(vertical, north, east);
        mInferenceRequest.set_input_tensor(mInputTensor);
        mInferenceRequest.infer();
        const auto &result = mInferenceRequest.get_output_tensor();  
        auto dPtr = reinterpret_cast<const float *> (result.data());
        probability->resize(MINIMUM_SIGNAL_LENGTH);
        std::copy(dPtr, dPtr + MINIMUM_SIGNAL_LENGTH, probability->data());
    } 
///private:
    const ov::Shape mInputShape{1, 3, MINIMUM_SIGNAL_LENGTH};
    const ov::element::Type mInputType{ov::element::f32};
    ov::Tensor mInputTensor{mInputType, mInputShape};
    ov::Core mCore;
    ov::CompiledModel mCompiledModel;
    ov::InferRequest mInferenceRequest;
    std::filesystem::path mFileName;
    std::string mDevice{"CPU"};
};
}
#else
class OpenVINO Impl
{
public:
};
#endif
