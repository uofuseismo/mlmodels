#include <torch/torch.h>

//using namespace UUSS::ForwardSimulation::Eikonet;

struct EikonetNetwork : torch::nn::Module
{
    // C'tor
    EikonetNetwork() :
        mELUOptions(torch::nn::functional::ELUFuncOptions().
                    alpha(0.42).inplace(true)),
        fc0(torch::nn::LinearOptions({6,   32}).bias(true)),
        fc1(torch::nn::LinearOptions({32, 512}).bias(true)),
        fc8(torch::nn::LinearOptions({512, 32}).bias(true)),
        fc9(torch::nn::LinearOptions({32,   1}).bias(true))
    {
        for (size_t layer = 0; layer < mLayers; ++layer)
        {
            torch::nn::Linear fc{torch::nn::LinearOptions({512, 512})
                                 .bias(true)};
            resnetFullyConnected1.push_back(std::move(fc));
        }
        for (size_t layer = 0; layer < mLayers; ++layer)
        {
//            resnetFullyConnected2.push_back
//                (torch::nn::LinearOptions({512, 512}).bias(true));
        }
        for (size_t layer = 0; layer < mLayers; ++layer)
        {
//            resnetFullyConnected3.push_back
//                (torch::nn::LinearOptions({512, 512}).bias(true));
        } 
      /* 
        // 1
        rnFC1(torch::nn::Conv1dOptions({1, 32, 21})
           .stride(1).padding(10).bias(true).dilation(1)),
        rnFC2(torch::nn::BatchNormOptions(32)
            .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 2
        conv2(torch::nn::Conv1dOptions({32, 64, 15})
           .stride(1).padding(7).bias(true).dilation(1)),
        batch2(torch::nn::BatchNormOptions(64)
            .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 3
        conv3(torch::nn::Conv1dOptions({64, 128, 11})
           .stride(1).padding(5).bias(true).dilation(1)),
        batch3(torch::nn::BatchNormOptions(128)
            .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 4
        fcn1(torch::nn::LinearOptions({6400, 512}).bias(true)),
        batch4(torch::nn::BatchNormOptions(512)
            .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 5
        fcn2(torch::nn::LinearOptions({512, 512}).bias(true)),
        batch5(torch::nn::BatchNormOptions(512)
            .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 6
        fcn3(torch::nn::LinearOptions({512, 3}).bias(true))
*/
    }

    torch::Tensor forward(const torch::Tensor &xIn)
    {
        auto x = torch::nn::functional::elu(fc0->forward(xIn), mELUOptions);
        x = torch::nn::functional::elu(fc1->forward(x), mELUOptions);
        for (size_t i = 0; i < resnetFullyConnected2.size(); ++i)
        {
            auto fcx0 = resnetFullyConnected3[i]->forward(x);
            x = torch::nn::functional::elu(resnetFullyConnected1[i]->forward(x), mELUOptions);
            x = torch::nn::functional::elu(
                  resnetFullyConnected2[i]->forward(x) + fcx0, mELUOptions);
        }
        x = torch::nn::functional::elu(fc8->forward(x), mELUOptions);
        auto tau = at::abs(fc9->forward(x));
        return tau;
    }
  
    torch::nn::functional::ELUFuncOptions mELUOptions;
    torch::nn::Linear fc0{nullptr};
    torch::nn::Linear fc1{nullptr};
    std::vector<torch::nn::Linear> resnetFullyConnected1;
    std::vector<torch::nn::Linear> resnetFullyConnected2;
    std::vector<torch::nn::Linear> resnetFullyConnected3;
    //torch::nn::ModuleList resnetFullyConnected3;
    torch::nn::Linear fc8{nullptr};
    torch::nn::Linear fc9{nullptr};
    size_t mLayers = 10;
};
