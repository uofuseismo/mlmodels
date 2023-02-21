#ifdef WITH_OPENVINO
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>
#include "../../../detectors/uNetOneComponentP/inference/utilities.hpp"
#define MODEL_NAME "PCNNNet"
namespace
{

template<class T = float>
struct Weights
{
    void loadFromHDF5(const std::string &fileName)
    {
        if (!std::filesystem::exists(fileName))
        {
            throw std::runtime_error("File name: " + fileName
                                   + " does not exist");
        }
        HDF5Loader dataLoader(fileName);
        dataLoader.openGroup("/model_weights/");
        auto dataSetNames = dataLoader.getDataSetsInGroup();
        for (const auto &dataSetName : dataSetNames)
        {
            std::vector<hsize_t> dimensions;
            std::vector<float> values;
            dataLoader.readDataSet(dataSetName, &dimensions, &values);
            //std::cout << dataSetName << " " << dimensions.size() << " " << values.size() << std::endl; 
            insert(dataSetName, values);
        }
    }
    void insert(const std::string &name,
                const std::vector<float> &weights)
    {
        if (contains(name))
        {
            throw std::invalid_argument(name + " already exists");
        }
        if (weights.empty())
        {
            throw std::invalid_argument("Weights are empty");
        }
        mWeights.insert(std::pair {name, weights});
    }
    /// @result All the 
    [[nodiscard]] std::vector<std::string> getNames() const noexcept
    {
        std::vector<std::string> result;
        result.reserve(mWeights.size());
        for (const auto &weight : mWeights)
        {
            result.push_back(weight.first);
        }
        return result;
    }
    [[nodiscard]] bool contains(const std::string &key) noexcept
    {
        return mWeights.contains(key);
    }
    const std::vector<T>& operator()(const std::string &key) const
    {
        return mWeights.at(key);
    }
///private:
    std::map<std::string, std::vector<T>> mWeights;
};

void createConvolutionLayer(
    const int layer,
    const unsigned long nFilters,
    const unsigned long nChannels,
    const unsigned long filterLength, 
    const Weights<float> &weights,
    const ov::Output<ov::Node> &input,
    std::shared_ptr<ov::opset10::Constant>    &convolutionConstantNode,
    std::shared_ptr<ov::opset10::Convolution> &convolutionNode,
    std::shared_ptr<ov::opset10::Constant>    &addConstantNode,
    std::shared_ptr<ov::opset10::Add>         &addNode,
    std::shared_ptr<ov::opset10::Relu>        &reluNode,
    std::shared_ptr<ov::opset10::Constant>    &batchNormConstantGammaNode,
    std::shared_ptr<ov::opset10::Constant>    &batchNormConstantBetaNode,
    std::shared_ptr<ov::opset10::Constant>    &batchNormConstantMeanNode,
    std::shared_ptr<ov::opset10::Constant>    &batchNormConstantVarianceNode,
    std::shared_ptr<ov::opset10::BatchNormInference> &batchNormNode,
    std::shared_ptr<ov::opset1::MaxPool>      &maxPoolNode)
{
    // NCW
    const ov::Shape convolutionShape({nFilters, nChannels, filterLength});
    const ov::Shape addShape({1, nFilters, 1});
    auto padding = static_cast<long> (filterLength/2);
    std::vector<ptrdiff_t> padBegin{padding};
    std::vector<ptrdiff_t> padEnd{padding};

    const ov::Strides convolutionStride{1};
    const ov::Strides convolutionDilation{1};
    const ov::Strides maxPoolStride{2};
    const ov::Strides maxPoolKernel{2};
    const ov::Shape maxPoolPadBegin{0};
    const ov::Shape maxPoolPadEnd{0};
    constexpr double epsilon{1.e-5};

    // Convolve
    convolutionConstantNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           convolutionShape,
           weights("conv1d_" + std::to_string(layer) + ".weight").data()
          );
    convolutionNode
        = std::make_shared<ov::opset10::Convolution>
          (
           input,
           convolutionConstantNode->output(0),
           convolutionStride,
           ov::CoordinateDiff(padBegin),
           ov::CoordinateDiff(padEnd),
           convolutionDilation
          );

    // Add bias
    addConstantNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           addShape,
           weights("conv1d_" + std::to_string(layer) + ".bias").data()
          );
    addNode
        = std::make_shared<ov::opset10::Add>
          (
           convolutionNode->output(0),
           addConstantNode->output(0)
          );

    // ReLU
    reluNode = std::make_shared<ov::opset10::Relu> (addNode->output(0));
   
    // Batch normalization
    const ov::Shape batchNormShape({nFilters});
    batchNormConstantGammaNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(layer) + ".weight").data()
          );  
    batchNormConstantBetaNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(layer) + ".bias").data()
          );
    batchNormConstantMeanNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(layer) + ".running_mean").data()
          );
    batchNormConstantVarianceNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(layer) + ".running_var").data()
          );
    batchNormNode
       = std::make_shared<ov::opset10::BatchNormInference>
         (
          reluNode->output(0),
          batchNormConstantGammaNode->output(0),
          batchNormConstantBetaNode->output(0),
          batchNormConstantMeanNode->output(0),
          batchNormConstantVarianceNode->output(0),
          epsilon
         );

    maxPoolNode
        = std::make_shared<ov::opset1::MaxPool>
          (
           batchNormNode->output(0),
           maxPoolStride,
           maxPoolPadBegin,
           maxPoolPadEnd,
           maxPoolKernel,
           ov::op::RoundingType::FLOOR,
           ov::op::PadType::EXPLICIT
          );
}

void createFullyConnectedLayer(
    const int fullyConnectedLayer,
    const int batchCounter,
    const unsigned long inputShape,
    const unsigned long outputShape,
    const Weights<float> &weights,
    const ov::Output<ov::Node> &input,
    std::shared_ptr<ov::opset10::Constant>  &fullyConnectedConstantNode,
    std::shared_ptr<ov::opset10::MatMul>    &fullyConnectedNode,
    std::shared_ptr<ov::opset10::Constant>  &addConstantNode,
    std::shared_ptr<ov::opset10::Add>       &addNode,
    std::shared_ptr<ov::opset10::Relu>      &reluNode,
    std::shared_ptr<ov::opset10::Constant>  &batchNormConstantGammaNode,
    std::shared_ptr<ov::opset10::Constant>  &batchNormConstantBetaNode,
    std::shared_ptr<ov::opset10::Constant>  &batchNormConstantMeanNode,
    std::shared_ptr<ov::opset10::Constant>  &batchNormConstantVarianceNode,
    std::shared_ptr<ov::opset10::BatchNormInference> &batchNormNode)
{
    constexpr double epsilon{1.e-5};
    constexpr bool transposeA{false};
    constexpr bool transposeB{false};
    // Fully connected Layer
    const ov::Shape fullyConnectedShape{inputShape, outputShape};
    fullyConnectedConstantNode
        = std::make_shared<ov::opset10::Constant> 
          (
           ov::element::Type_t::f32,
           fullyConnectedShape,
           weights("fcn_" + std::to_string(fullyConnectedLayer)
                 + ".weight").data()
         );
    fullyConnectedNode
         = std::make_shared<ov::opset10::MatMul>
           (
            input,
            fullyConnectedConstantNode->output(0),
            transposeA,
            transposeB
           );

    // Add bias
    const ov::Shape addShape({1, outputShape});
    addConstantNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           addShape,
           weights("fcn_" + std::to_string(fullyConnectedLayer) + ".bias").data()
          );
    addNode
        = std::make_shared<ov::opset10::Add>
          (
           fullyConnectedNode->output(0),
           addConstantNode->output(0)
          );

    // ReLU
    reluNode = std::make_shared<ov::opset10::Relu> (addNode->output(0));

    // Batch normalization
    const ov::Shape batchNormShape({outputShape});
    batchNormConstantGammaNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(batchCounter) + ".weight").data()
          );
    batchNormConstantBetaNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(batchCounter) + ".bias").data()
          );
    batchNormConstantMeanNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(batchCounter)
                 + ".running_mean").data()
          );
    batchNormConstantVarianceNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           batchNormShape,
           weights("bn_" + std::to_string(batchCounter)
                 + ".running_var").data()
          );
    batchNormNode
       = std::make_shared<ov::opset10::BatchNormInference>
         (
          reluNode->output(0),
          batchNormConstantGammaNode->output(0),
          batchNormConstantBetaNode->output(0),
          batchNormConstantMeanNode->output(0),
          batchNormConstantVarianceNode->output(0),
          epsilon
         );
}

void createFinalLayer(
    const int fullyConnectedLayer,
    const unsigned long inputShape,
    const unsigned long outputShape,
    const Weights<float> &weights,
    const ov::Output<ov::Node> &input,
    std::shared_ptr<ov::opset10::Constant>  &fullyConnectedConstantNode,
    std::shared_ptr<ov::opset10::MatMul>    &fullyConnectedNode,
    std::shared_ptr<ov::opset10::Constant>  &addConstantNode,
    std::shared_ptr<ov::opset10::Add>       &addNode,
    std::shared_ptr<ov::opset10::Softmax>   &clampNode)
{
    constexpr bool transposeA{false};
    constexpr bool transposeB{false};
    // Fully connected Layer
    const ov::Shape fullyConnectedShape{inputShape, outputShape};
    fullyConnectedConstantNode
        = std::make_shared<ov::opset10::Constant> 
          (
           ov::element::Type_t::f32,
           fullyConnectedShape,
           weights("fcn_" + std::to_string(fullyConnectedLayer)
                 + ".weight").data()
         );
    fullyConnectedNode
         = std::make_shared<ov::opset10::MatMul>
           (
            input,
            fullyConnectedConstantNode->output(0),
            transposeA,
            transposeB
           );

    // Add bias
    const ov::Shape addShape({1, outputShape});
    addConstantNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           addShape,
           weights("fcn_" + std::to_string(fullyConnectedLayer) + ".bias").data()
          );
    addNode
        = std::make_shared<ov::opset10::Add>
          (
           fullyConnectedNode->output(0),
           addConstantNode->output(0)
          );

    // Clampm 
    clampNode
        = std::make_shared<ov::opset10::Clamp> (addNode->output(0), -50, 50);
}

std::shared_ptr<ov::Model>
    createModel(const Weights<float> &weights,
                const uint64_t batchSize = 1)
{
    // Architecture:
    //  Layer 1: Convolve -> Add Bias -> ReLU -> BatchNorm -> MaxPool
    //  Layer 2: Convolve -> Add Bias -> ReLU -> BatchNorm -> MaxPool
    //  Layer 3: Convolve -> Add Bias -> ReLU -> BatchNorm -> MaxPool
    //  Layer 4: Flatten
    //  Layer 5: MatrixMultiply (FullyConnected) -> ReLU -> BatchNorm
    //  Layer 6: MatrixMultiply (FullyConnected) -> ReLU -> BatchNorm
    //  Layer 6: MatrixMultiply (FullyConnected) -> Clamp

    // Input data.  OpenVINO defines a parameter as something the user will give
    // the model.   Data is NCW
    auto parameterNode
        = std::make_shared<ov::opset10::Parameter>
            (ov::element::Type_t::f32,
             ov::Shape({batchSize, N_CHANNELS, EXPECTED_SIGNAL_LENGTH}));
/*
    auto parameterNode
        = std::make_shared<ov::opset10::Parameter>
            (ov::element::Type_t::f32,
             ov::PartialShape({ov::Dimension::dynamic(),
                               N_CHANNELS,
                               EXPECTED_SIGNAL_LENGTH}));
*/
    // Layer 1
    std::shared_ptr<ov::opset10::Constant>    firstConvolutionConstantNode;
    std::shared_ptr<ov::opset10::Convolution> firstConvolutionNode;
    std::shared_ptr<ov::opset10::Constant>    firstAddConstantNode;
    std::shared_ptr<ov::opset10::Add>         firstAddNode;
    std::shared_ptr<ov::opset10::Relu>        firstReluNode;
    std::shared_ptr<ov::opset10::Constant> firstBatchNormConstantGammaNode;
    std::shared_ptr<ov::opset10::Constant> firstBatchNormConstantBetaNode;
    std::shared_ptr<ov::opset10::Constant> firstBatchNormConstantMeanNode;
    std::shared_ptr<ov::opset10::Constant> firstBatchNormConstantVarianceNode;
    std::shared_ptr<ov::opset10::BatchNormInference> firstBatchNormNode;
    std::shared_ptr<ov::opset1::MaxPool>  firstMaxPoolNode;
 
    createConvolutionLayer(1, 32, N_CHANNELS, 21,
                           weights,
                           parameterNode->output(0),
                           firstConvolutionConstantNode,
                           firstConvolutionNode,
                           firstAddConstantNode,
                           firstAddNode,
                           firstReluNode,
                           firstBatchNormConstantGammaNode,
                           firstBatchNormConstantBetaNode,
                           firstBatchNormConstantMeanNode,
                           firstBatchNormConstantVarianceNode,
                           firstBatchNormNode,
                           firstMaxPoolNode);

    // Layer 2
    std::shared_ptr<ov::opset10::Constant>    secondConvolutionConstantNode;
    std::shared_ptr<ov::opset10::Convolution> secondConvolutionNode;
    std::shared_ptr<ov::opset10::Constant>    secondAddConstantNode;
    std::shared_ptr<ov::opset10::Add>         secondAddNode;
    std::shared_ptr<ov::opset10::Relu>        secondReluNode;
    std::shared_ptr<ov::opset10::Constant> secondBatchNormConstantGammaNode;
    std::shared_ptr<ov::opset10::Constant> secondBatchNormConstantBetaNode;
    std::shared_ptr<ov::opset10::Constant> secondBatchNormConstantMeanNode;
    std::shared_ptr<ov::opset10::Constant> secondBatchNormConstantVarianceNode;
    std::shared_ptr<ov::opset10::BatchNormInference> secondBatchNormNode;
    std::shared_ptr<ov::opset1::MaxPool>  secondMaxPoolNode;

    createConvolutionLayer(2, 64, 32, 15,
                           weights,
                           firstMaxPoolNode->output(0),
                           secondConvolutionConstantNode,
                           secondConvolutionNode,
                           secondAddConstantNode,
                           secondAddNode,
                           secondReluNode,
                           secondBatchNormConstantGammaNode,
                           secondBatchNormConstantBetaNode,
                           secondBatchNormConstantMeanNode,
                           secondBatchNormConstantVarianceNode,
                           secondBatchNormNode,
                           secondMaxPoolNode);

    // Layer 3
    std::shared_ptr<ov::opset10::Constant>    thirdConvolutionConstantNode;
    std::shared_ptr<ov::opset10::Convolution> thirdConvolutionNode;
    std::shared_ptr<ov::opset10::Constant>    thirdAddConstantNode;
    std::shared_ptr<ov::opset10::Add>         thirdAddNode;
    std::shared_ptr<ov::opset10::Relu>        thirdReluNode;
    std::shared_ptr<ov::opset10::Constant> thirdBatchNormConstantGammaNode;
    std::shared_ptr<ov::opset10::Constant> thirdBatchNormConstantBetaNode;
    std::shared_ptr<ov::opset10::Constant> thirdBatchNormConstantMeanNode;
    std::shared_ptr<ov::opset10::Constant> thirdBatchNormConstantVarianceNode;
    std::shared_ptr<ov::opset10::BatchNormInference> thirdBatchNormNode;
    std::shared_ptr<ov::opset1::MaxPool>  thirdMaxPoolNode;

    createConvolutionLayer(3, 128, 64, 11,
                           weights,
                           secondMaxPoolNode->output(0),
                           thirdConvolutionConstantNode,
                           thirdConvolutionNode,
                           thirdAddConstantNode,
                           thirdAddNode,
                           thirdReluNode,
                           thirdBatchNormConstantGammaNode,
                           thirdBatchNormConstantBetaNode,
                           thirdBatchNormConstantMeanNode,
                           thirdBatchNormConstantVarianceNode,
                           thirdBatchNormNode,
                           thirdMaxPoolNode);

    // Reshape?
    std::vector<int64_t> newShape{6400};
    const ov::Shape flattenDimensions{newShape.size()};
    auto reshapeConstantNode = std::make_shared<ov::opset10::Constant> (ov::element::i64, flattenDimensions, newShape);
    auto reshapeNode
        = std::make_shared<ov::opset1::Reshape> (thirdMaxPoolNode->output(0),
                                                 reshapeConstantNode->output(0),
                                                 true);

    // Fully connected layer 1
    std::shared_ptr<ov::opset10::Constant>  firstFullyConnectedConstantNode;
    std::shared_ptr<ov::opset10::MatMul>    firstFullyConnectedNode;
    std::shared_ptr<ov::opset10::Constant>  fourthAddConstantNode;
    std::shared_ptr<ov::opset10::Add>       fourthAddNode;
    std::shared_ptr<ov::opset10::Relu>      fourthReluNode;
    std::shared_ptr<ov::opset10::Constant>  fourthBatchNormConstantGammaNode;
    std::shared_ptr<ov::opset10::Constant>  fourthBatchNormConstantBetaNode;
    std::shared_ptr<ov::opset10::Constant>  fourthBatchNormConstantMeanNode;
    std::shared_ptr<ov::opset10::Constant>  fourthBatchNormConstantVarianceNode;
    std::shared_ptr<ov::opset10::BatchNormInference> fourthBatchNormNode;

    createFullyConnectedLayer(1, 4, 6400, 512,
                              weights,
                              reshapeNode->output(0),
                              firstFullyConnectedConstantNode,
                              firstFullyConnectedNode,
                              fourthAddConstantNode,
                              fourthAddNode,
                              fourthReluNode,
                              fourthBatchNormConstantGammaNode,
                              fourthBatchNormConstantBetaNode,
                              fourthBatchNormConstantMeanNode,
                              fourthBatchNormConstantVarianceNode,
                              fourthBatchNormNode);

    // Fully connected layer 2
    std::shared_ptr<ov::opset10::Constant>  secondFullyConnectedConstantNode;
    std::shared_ptr<ov::opset10::MatMul>    secondFullyConnectedNode;
    std::shared_ptr<ov::opset10::Constant>  fifthAddConstantNode;
    std::shared_ptr<ov::opset10::Add>       fifthAddNode;
    std::shared_ptr<ov::opset10::Relu>      fifthReluNode;
    std::shared_ptr<ov::opset10::Constant>  fifthBatchNormConstantGammaNode;
    std::shared_ptr<ov::opset10::Constant>  fifthBatchNormConstantBetaNode;
    std::shared_ptr<ov::opset10::Constant>  fifthBatchNormConstantMeanNode;
    std::shared_ptr<ov::opset10::Constant>  fifthBatchNormConstantVarianceNode;
    std::shared_ptr<ov::opset10::BatchNormInference> fifthBatchNormNode;

    createFullyConnectedLayer(2, 5, 512, 512,
                              weights,
                              fourthBatchNormNode->output(0),
                              secondFullyConnectedConstantNode,
                              secondFullyConnectedNode,
                              fifthAddConstantNode,
                              fifthAddNode,
                              fifthReluNode,
                              fifthBatchNormConstantGammaNode,
                              fifthBatchNormConstantBetaNode,
                              fifthBatchNormConstantMeanNode,
                              fifthBatchNormConstantVarianceNode,
                              fifthBatchNormNode);

    // Last layer is a one-off
    std::shared_ptr<ov::opset10::Constant>  thirdFullyConnectedConstantNode;
    std::shared_ptr<ov::opset10::MatMul>    thirdFullyConnectedNode;
    std::shared_ptr<ov::opset10::Constant>  sixthAddConstantNode;
    std::shared_ptr<ov::opset10::Add>       sixthAddNode;
    std::shared_ptr<of::opset10::Clamp>     clampNode;

    createFinalLayer(3, 512, 1,
                     weights,
                     fifthBatchNormNode->output(0),
                     thirdFullyConnectedConstantNode,
                     thirdFullyConnectedNode,
                     sixthAddConstantNode,
                     sixthAddNode,
                     clampNode);
    // Result
    clampNode->get_output_tensor(0).set_names({"output_signal"});
    auto result
        = std::make_shared<ov::opset10::Result> (clampNode->output(0));

    std::shared_ptr<ov::Model> modelPointer
        = std::make_shared<ov::Model> (result,
                                       ov::ParameterVector{parameterNode},
                                       MODEL_NAME);

    return modelPointer;
}

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
    /// @brief Create from weights
    void createFromWeights(const Weights<float> &weights,
                           const size_t batchSize = 1)
    {
       auto model = createModel(weights, batchSize);
       mCompiledModel = mCore.compile_model(model, mDevice);
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
    /// @brief Sets the data
    template<typename T>
    void setSignal(const std::vector<T> &vertical)
    {
        if (vertical.size() != EXPECTED_SIGNAL_LENGTH)
        {
            throw std::invalid_argument("Vertical is wrong size");
        }
        auto data = reinterpret_cast<float *> (mInputTensor.data());
        ::rescaleAndCopy(EXPECTED_SIGNAL_LENGTH, vertical.data(), data);
    }
    /// @brief Perform inference
    template<typename T>
    T predict(const std::vector<T> &vertical)
    {
        mInferenceRequest = mCompiledModel.create_infer_request();
        setSignal(vertical);
        mInferenceRequest.set_input_tensor(mInputTensor);
        mInferenceRequest.infer();
        const auto &result = mInferenceRequest.get_output_tensor();  
        auto pickCorrection = reinterpret_cast<const float *> (result.data());
        return std::tuple {static_cast<T> (pickCorrection)};
    } 
///private:
    const ov::Shape mInputShape{1,  1, EXPECTED_SIGNAL_LENGTH};
    const ov::element::Type mInputType{ov::element::f32};
    ov::Tensor mInputTensor{mInputType, mInputShape};
    ov::Core mCore;
    ov::CompiledModel mCompiledModel;
    ov::InferRequest mInferenceRequest;
    std::filesystem::path mFileName;
    std::string mDevice{"CPU"};
};

}
#endif
