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
        const Inference::Device requestedDevice = Inference::Device::CPU)
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
    }
    /// @brief Sets the data
    template<typename T>
    void setSignals(const std::vector<T> &vertical,
                    const std::vector<T> &north,
                    const std::vector<T> &east)
    {
        if (vertical.size() != EXPECTED_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("Vertical is wrong size");
        }
        if (north.size() != EXPECTED_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("North is wrong size");
        }
        if (east.size() != EXPECTED_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("East is wrong size");
        }
        auto data = reinterpret_cast<float *> (mInputTensor.data());
        ::rescaleAndCopy(EXPECTED_SIGNAL_LENGTH,
                         vertical.data(), north.data(), east.data(), data);
    }
    /// @brief Perform inference
    template<typename T>
    void predictProbability(const std::vector<T> &vertical,
                            const std::vector<T> &north,
                            const std::vector<T> &east,
                            std::vector<T> *probability)
    {
        mInferenceRequest = mCompiledModel.create_infer_request();
        setSignals(vertical, north, east);
        mInferenceRequest.set_input_tensor(mInputTensor);
        mInferenceRequest.infer();
        const auto &result = mInferenceRequest.get_output_tensor();  
        auto pPtr = reinterpret_cast<const float *> (result.data());
        probability->resize(EXPECTED_SIGNAL_LENGTH);
        std::copy(pPtr, pPtr + EXPECTED_SIGNAL_LENGTH, probability->data());
    } 
    /// @brief Perform inference in batches
    template<typename T>
    void predictProbabilitySlidingWindow(const std::vector<T> &vertical,
                                         const std::vector<T> &north,
                                         const std::vector<T> &east,
                                         std::vector<T> *probability,
                                         const int windowStart = 254, // 1008/2 - 250
                                         const int windowEnd   = 754) // 1008/2 + 250
    {
        constexpr int nComponents{3};
        auto batchSize = static_cast<int> (mBatchInputShape[0]);
        auto nSamples = static_cast<int> (vertical.size());
#ifndef NDEBUG
        assert(nSamples >= EXPECTED_SIGNAL_LENGTH);
#endif
        probability->resize(nSamples, 0); // Initialize result
        auto batchData = reinterpret_cast<float *> (mBatchInputTensor.data());
        // We advance the window by the window size
        int windowSize = windowEnd - windowStart;
        int lastCopy = 0;
        // Start the sliding window at iWindowStart
        for (int k = 0; k < nSamples; k = k + windowSize*batchSize)
        {
            // Extract source signals and store in sliding window
            int nWindows = 0;
            for (int batch = 0; batch < batchSize; ++batch)
            {
                // Extract the source signal from [i1:i2]
                int i1 = k  + batch*windowSize;
                int i2 = i1 + EXPECTED_SIGNAL_LENGTH;
                int j1 = batch*(nComponents*EXPECTED_SIGNAL_LENGTH);
                // Don't go out of bounds 
                if (i2 > nSamples)
                {
                    // Make sure mBatchInputTensor is filled with some value.
                    // We'll use zero but this will be ignored when we
                    // extract the probability signal.
                    int j2 = batchSize*(nComponents*EXPECTED_SIGNAL_LENGTH);
                    std::fill(batchData + j1, batchData + j2, 0);
                    break;
                }
                //std::cout << i1 << " [" << i1 + windowStart << " " << i1 + windowEnd << "] " << i2 << " " << j1 << " " << std::endl;
                // Perform normalization
                ::rescaleAndCopy(EXPECTED_SIGNAL_LENGTH,
                                 vertical.data() + i1,
                                 north.data() + i1,
                                 east.data() + i1,
                                 batchData + j1);
                nWindows = nWindows + 1;
            }
            // Perform inference on batch
            mInferenceRequest = mCompiledModel.create_infer_request();
            mInferenceRequest.set_input_tensor(mBatchInputTensor);
            mInferenceRequest.infer(); // Blocking
            // Pick out result for all windows
            const auto &result = mInferenceRequest.get_output_tensor();
            auto pPtr = reinterpret_cast<const float *> (result.data());
            for (int window = 0; window < nWindows; ++window)
            {
                // Copy probabilities in the window into output vector
                int i1 = window*EXPECTED_SIGNAL_LENGTH + windowStart;
                int i2 = i1 + windowSize;//(EXPECTED_SIGNAL_LENGTH - windowStart); //windowSize;
                int j1 = k + window*windowSize + windowStart;
                // First window starts at 0 since there's nothing to overwrite
                if (k == 0 && window == 0)
                {
                    i1 = 0;
                    i2 = windowStart + windowSize;
                    j1 = 0;
                }
#ifndef NDEBUG
                assert(j1 + (EXPECTED_SIGNAL_LENGTH - windowSize) <= nSamples);
#endif
                //std::cout << j1 << " " << j1 + (i2 - i1) << std::endl;
                lastCopy = j1 + windowSize;
                std::copy(pPtr + i1, pPtr + i2, probability->data() + j1);
            }
        }
        // Do last window?  It's possible we already did it if j1 == nSamples
        //std::cout << "LastCopy: " << lastCopy << std::endl;
        int leftOverSignal = nSamples - lastCopy;
        if (leftOverSignal > 0 && nSamples - EXPECTED_SIGNAL_LENGTH > 0)
        {
            // Just do inference on the last EXPECTED_SIGNAL_LENGTH segment
            int i1 = nSamples - EXPECTED_SIGNAL_LENGTH; 
            // Perform normalization
            auto inputData = reinterpret_cast<float *> (mInputTensor.data());
            ::rescaleAndCopy(EXPECTED_SIGNAL_LENGTH,
                             vertical.data() + i1,
                             north.data() + i1,
                             east.data() + i1,
                             inputData);
            // Apply the model
            mInferenceRequest = mCompiledModel.create_infer_request();
            mInferenceRequest.set_input_tensor(mInputTensor);
            mInferenceRequest.infer(); // Blocking
            // Extract result
            const auto &result = mInferenceRequest.get_output_tensor();
            auto pPtr = reinterpret_cast<const float *> (result.data());
            i1 = EXPECTED_SIGNAL_LENGTH - leftOverSignal;
            int i2 = EXPECTED_SIGNAL_LENGTH;
            int j1 = lastCopy;
#ifndef NDEBUG
            int j2 = nSamples;
            assert(i1 >= 0);
            assert(i2 - i1 == j2 - j1);
            assert(i2 - i1 <= EXPECTED_SIGNAL_LENGTH);
#endif
            //std::cout << i1 << " " << i2 << " " << j1 << " " << nSamples << std::endl;
            std::copy(pPtr + i1, pPtr + i2, probability->data() + j1);
        }
        
    }
///private:
    const ov::Shape mBatchInputShape{16, 3, EXPECTED_SIGNAL_LENGTH};
    const ov::Shape mInputShape{1,  3, EXPECTED_SIGNAL_LENGTH};
    const ov::element::Type mInputType{ov::element::f32};
    ov::Tensor mBatchInputTensor{mInputType, mBatchInputShape};
    ov::Tensor mInputTensor{mInputType, mInputShape};
    ov::Core mCore;
    ov::CompiledModel mCompiledModel;
    ov::InferRequest mInferenceRequest;
    std::filesystem::path mFileName;
    std::string mDevice{"CPU"};
};
}
#else
class OpenVINOImpl
{
public:
};
#endif
