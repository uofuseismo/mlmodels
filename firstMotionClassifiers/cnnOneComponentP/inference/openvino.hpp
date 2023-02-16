#ifdef WITH_OPENVINO
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset10.hpp>
#define MODEL_NAME "FMNet"
namespace
{

struct Coefficients
{
    std::vector<float> mConvolution1;
    std::vector<float> mConvolutionBias1;

    std::vector<float> mConvolution2;
    std::vector<float> mConvolutionBias2;

    std::vector<float> mConvolution3;
    std::vector<float> mConvolutionBias3;

    std::vector<float> mFullyConnected1;
    std::vector<float> mFullyConnectedBias1;

    std::vector<float> mFullyConnected2;
    std::vector<float> mFullyConnectedBias2;

    std::vector<float> mFullyConnected3;
    std::vector<float> mFullyConnectedBias3;
};

std::shared_ptr<ov::Model>
    createModel(const Coefficients &coefficients,
                const size_t batchSize = 1)
{
    const ov::Strides convolutionStride{1};
    const ov::Strides convolutionDilitation{1};
    const ov::Strides maxPoolStride{2};
    const ov::Strides maxPoolKernel{2};
    constexpr double epsilon{1.e-5};
    constexpr bool transposeA{false};
    constexpr bool transposeB{false};
    // Input data.  OpenVINO defines a parameter as something the user will give
    // the model. 
    auto parameterNode
        = std::make_shared<ov::opset10::Parameter>
            (ov::element::Type_t::f32,
             ov::PartialShape({ov::Dimension::dynamic(),
                               N_CHANNELS,
                               EXPECTED_SIGNAL_LENGTH}));

    // Convolution 1 (32 filters each of length 21)
    const ov::Shape firstConvolutionShape({32, 1, 21});
    auto firstConvolutionConstantNode
        = std::make_shared<ov::opset10::Constant>
          (
           ov::element::Type_t::f32,
           firstConvolutionShape,
           coefficients.mConvolution1.data()
          );
    auto firstConvolutionNode
        = std::make_shared<ov::opset10::Convolution>
          (
           parameterNode->output(0),
           firstConvolutionConstantNode->output(0),
           convolutionStride,
           ov::CoordinateDiff(10), // Pad begin
           ov::CoordinateDiff(10), // Pad end
           convolutionDilitation
          );

    // Convolution 2 (64 filters each of length 15)
    const ov::Shape secondConvolutionShape({64, 32, 15});

    // Convolution 3 (128 filters each of length 11)
    const ov::Shape thirdConvolutionShape({128, 64, 15});


/*
    // First fully connected Layer
    const ov::Shape firstFullyConnectedShape{6400, 512};
    auto firstFullyConnectedConstantNode
        = std::make_shared<opset8::Constant> (element::Type_t::f32,
                                              firstFullyConnectedShape,
                                              coefficients.mFullyConnected1.data());

    // Second fully connected layer
    const ov::Shape secondFullyConnectedShape{512, 512};
    auto secondFullyConnectedConstantNode
        = std::make_shared<opset8::Constant> (element::Type_t::f32,
                                              secondFullyConnectedShape,
                                              coefficients.mFullyConnected2.data());
    auto fullyConnectedSecondNode
         = std::make_shared<opset10::MatMul>(firstFullyConnectedConstantNode->output(0),
                                             secondFullyConnectedConstantNode->output(0),
                                             transposeA,
                                             transposeB);
    const ov::Shape addSecondFullyConnectedShape{512, 1};
    

    // Third fully connected layer
    const ov::Shape thirdFullyConnectedShape{512, 3};
    auto thirdFullyConnectedConstantNode
        = std::make_shared<opset8::Constant> (element::Type_t::f32,
                                              thirdFullyConnectedShape,
                                              coefficients.mFullyConnected3.data());
    const ov::Shape addThirdFullyConnectedShape{3, 1};
    auto fourthFCNBiasConstantNode
        = std::make_shared<opset10::Constant> (ov::element::Type_t::f32,
                                               addThirdFCNShape,
                                               coeffients.mFCNBias3.data());
    auto fourthFCNBiasNode = std::make_shared<ov::opset10::Add> (matMulSecondNode->output(0), add4ConstantNode->output(0));
*/

    // Squashing function
    auto softMaxNode
        = std::make_shared<ov::opset10::Softmax> ();//add4Node->output(0), 1);
    softMaxNode->get_output_tensor(0).set_names({"output_signal"});
    // Result
    auto result = std::make_shared<ov::opset10::Result> (softMaxNode->output(0));

    std::shared_ptr<ov::Model> modelPointer
        = std::make_shared<ov::Model> (result,
                                       ov::ParameterVector{parameterNode},
                                       MODEL_NAME);

    return modelPointer;
}

}
#endif
