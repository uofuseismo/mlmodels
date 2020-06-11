#include <string>
#include <torch/torch.h>
#include "uuss/firstMotion/fmnet/model.hpp"
#include "private/loadHDF5Weights.hpp"

using namespace UUSS::FirstMotion::FMNet;

namespace
{
/*!
 * @brief Computes the absolute max value of an array.
 */
template<class T> T getMaxAbs(const int npts, const T *__restrict__ v)
{
#ifdef USE_PSTL
    auto result = std::minmax_element(std::execution::unseq, v, v+npts);
#else
    auto result = std::minmax_element(v, v+npts);
#endif
    auto amax = std::max( std::abs(*result.first), std::abs(*result.second) );
    return amax;
}
/*!
 * @brief Performs a min/max normalization and copies.
 * @param[in] npts     The number of points
 * @param[in] z        The vertical trace.  This has dimension [npts].
 * @param[in] n        The north trace.  This has dimension [npts].
 * @param[in] e        The east trae.  This has dimension [npts].
 * @param[out] tensor  The rescaled data for the neural network.
 *                     This has dimension [3 x npts] and is stored
 *                     row major format.
 * @retval True indicates that this is a live trace.
 * @retval False indicates that this is a dead trace.
 */
template<class T>
bool rescaleAndCopy(const int npts,
                    const T *__restrict__ z,
                    T *__restrict__ tensor)
{
    auto zmax = getMaxAbs(npts, z);
    if (zmax != 0)
    {
        auto xnorm = 1.0/zmax;
#ifdef USE_PSTL
        std::transform(std::execution::unseq,
                       z, z+npts, tensor,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
#else
        std::transform(z, z+npts, tensor,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
#endif
        return true;
    }
    else
    {
        std::fill(tensor, tensor+npts, 0);
        return false;
    }
}

}

struct FMNetwork : torch::nn::Module
{
    // C'tor
    FMNetwork() :
        // 1
        conv1(torch::nn::Conv1dOptions({1, 32, 21})
           .stride(1).padding(10).bias(true).dilation(1)),
        batch1(torch::nn::BatchNormOptions(32)
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
    {
        register_module("conv1d_1", conv1);
        register_module("bn_1", batch1);

        register_module("conv1d_2", conv2);
        register_module("bn_2", batch2);

        register_module("conv1d_3", conv3);
        register_module("bn_3", batch3);

        register_module("fcn_1", fcn1);
        register_module("bn_4", batch4);

        register_module("fcn_2", fcn2);
        register_module("bn_5", batch5);

        register_module("fcn_3", fcn3);
    }
    /// Forward
    torch::Tensor forward(const torch::Tensor &xIn)
    {
        constexpr int64_t mMaxPoolSize = 2;
        constexpr int64_t mPoolStride = 2;
//std::cout << conv1->weight << std::endl;
//std::cout << conv1->bias << std::endl;
        auto x = conv1->forward(xIn);
        x = torch::relu(x);
        x = batch1->forward(x);
std::cout << x << std::endl;
        x = torch::max_pool1d(x, mMaxPoolSize, mPoolStride);

        x = conv2->forward(x);
        x = torch::relu(x);
        x = batch2->forward(x);
        x = torch::max_pool1d(x, mMaxPoolSize, mPoolStride);

        x = conv3->forward(x);
        x = torch::relu(x);
        x = batch3->forward(x);
        x = torch::max_pool1d(x, mMaxPoolSize, mPoolStride);

        x = at::flatten(x, 1);

        x = fcn1->forward(x);
        x = torch::relu(x);
        x = batch4->forward(x);

        x = fcn2->forward(x);
        x = torch::relu(x);
        x = batch5->forward(x);

        x = fcn3->forward(x);
        return torch::softmax(x, 1);
    }
    /// Loads the weights from file
    void loadWeightsFromHDF5(const std::string &fileName,
                             const bool gpu = false,
                             const bool verbose = false)
    {
        HDF5Loader loader;
        loader.openFile(fileName);
        loader.openGroup("/model_weights/sequential_1"); 

        readWeightsAndBiasFromHDF5(loader, "conv1d_1", conv1, gpu, verbose);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_1", batch1,
                                              gpu, verbose);

        readWeightsAndBiasFromHDF5(loader, "conv1d_2", conv2, gpu, verbose);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_2", batch2,
                                              gpu, verbose);

        readWeightsAndBiasFromHDF5(loader, "conv1d_3", conv3, gpu, verbose);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_3", batch3,
                                              gpu, verbose);


        readWeightsAndBiasFromHDF5(loader, "fcn_1", fcn1, gpu, verbose);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_4", batch4,
                                              gpu, verbose);

        readWeightsAndBiasFromHDF5(loader, "fcn_2", fcn2, gpu, verbose);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_5", batch5,
                                              gpu, verbose);

        readWeightsAndBiasFromHDF5(loader, "fcn_3", fcn3, gpu, verbose);

    }
//private:
    torch::nn::Conv1d conv1{nullptr};
    torch::nn::BatchNorm1d batch1{nullptr};
    torch::nn::Conv1d conv2{nullptr};
    torch::nn::BatchNorm1d batch2{nullptr};
    torch::nn::Conv1d conv3{nullptr};
    torch::nn::BatchNorm1d batch3{nullptr};
    torch::nn::Linear fcn1{nullptr};
    torch::nn::BatchNorm1d batch4{nullptr};
    torch::nn::Linear fcn2{nullptr};
    torch::nn::BatchNorm1d batch5{nullptr};
    torch::nn::Linear fcn3{nullptr};
};

/// Structure with implementation
template<UUSS::Device E>
class Model<E>::FMNetImpl
{
public:
    void toGPU()
    {
        if (!mOnGPU && mUseGPU && mHaveGPU)
        {
            mNetwork.to(torch::kCUDA);
            mOnGPU = true;
        }
    }
    FMNetwork mNetwork;
    double mPolarityThreshold = 1./3.;
    int mSignalLength = 400;
    bool mUseGPU = false;
    bool mOnGPU = false;
    bool mHaveCoefficients = false;
    bool mHaveGPU = torch::cuda::is_available();
};

/// C'tor
template<>
Model<UUSS::Device::CPU>::Model() :
    pImpl(std::make_unique<FMNetImpl> ())
{
    pImpl->mNetwork.eval();
}

/// C'tor
template<>
Model<UUSS::Device::GPU>::Model () :
    pImpl(std::make_unique<FMNetImpl> ())
{
    if (!torch::cuda::is_available())
    {
        std::cerr << "CUDA not available - using CPU" << std::endl;
    }
    else
    {
        pImpl->mUseGPU = true;
    }
    pImpl->toGPU();
    pImpl->mNetwork.eval();
}

/// Destructor
template<UUSS::Device E>
Model<E>::~Model() = default;

/// Load from weights
template<UUSS::Device E>
void Model<E>::loadWeightsFromHDF5(const std::string &fileName,
                                   const bool verbose)
{
    pImpl->mNetwork.loadWeightsFromHDF5(fileName, pImpl->mUseGPU, verbose);
    if (pImpl->mUseGPU){pImpl->toGPU();}
    pImpl->mNetwork.eval();
    pImpl->mHaveCoefficients = true;
}

/// Returns the input signal length
template<UUSS::Device E>
int Model<E>::getSignalLength() const noexcept
{
    return pImpl->mSignalLength;
}

/// Are the model coefficients set?
template<UUSS::Device E>
bool Model<E>::haveModelCoefficients() const noexcept
{
    return pImpl->mHaveCoefficients;
}

/// Set polarity threshold
template<UUSS::Device E>
void Model<E>::setPolarityThreshold(const double threshold)
{
    if (threshold < 0 || threshold > 1)
    {
        throw std::invalid_argument("threshold = " + std::to_string(threshold)
                                  + " must be in range [0,1]\n");
    }
    pImpl->mPolarityThreshold = threshold;
}

/// Get polarity threshold
template<UUSS::Device E>
double Model<E>::getPolarityThreshold() const noexcept
{
    return pImpl->mPolarityThreshold;
}

/// Predicts up/down/unkonwn
template<UUSS::Device E>
int Model<E>::predict(const int nSamples, const float z[]) const
{
    float pUp, pDown, pUnknown;
    predictProbability(nSamples, z, &pUp, &pDown, &pUnknown);
    auto thresh = static_cast<float> (getPolarityThreshold());
    if (pUp > pDown)
    {
        if (pUp > pUnknown && pUp > thresh){return 1;}
        return 0;
    }
    else
    {
        if (pDown > pUnknown && pDown > thresh){return -1;}
        return 0;
    }
}

/// Compute the probability of up/down/unknown on cpu
template<>
void Model<UUSS::Device::CPU>::predictProbability(
    const int nSamples, const float z[],
    float *pUp, float *pDown, float *pUnknown) const
{
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients\n");
    }
    if (nSamples != getSignalLength())
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must equal "
                                  + std::to_string(getSignalLength()) + "\n");
    }
    if (z == nullptr){throw std::invalid_argument("z is NULL\n");}
    if (pUp == nullptr){throw std::invalid_argument("pUp is NULL\n");}
    if (pDown == nullptr){throw std::invalid_argument("pDown is NULL\n");}
    if (pUnknown == nullptr){throw std::invalid_argument("pUnkown is NULL\n");}
    // Default a result
    *pUp = 0;
    *pDown = 0;
    *pUnknown = 1;
    auto X = torch::zeros({1, 1, 400},
                          torch::TensorOptions().dtype(torch::kFloat32)
                          .requires_grad(false));
    float *signalPtr = X.data_ptr<float> ();
    auto lAlive = rescaleAndCopy(pImpl->mSignalLength, z, signalPtr);
    if (!lAlive){return;} // Dead trace nothing more to do
    auto p = pImpl->mNetwork.forward(X);
    float *pPtr = p.data_ptr<float> ();
    *pUp = pPtr[0];
    *pDown = pPtr[1];
    *pUnknown = pPtr[2];
}

/// Compute the probability of up/down/unknown on gpu
template<>
void Model<UUSS::Device::GPU>::predictProbability(
    const int nSamples, const float z[],
    float *pUp, float *pDown, float *pUnknown) const
{
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients\n");
    }
    if (nSamples != getSignalLength())
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must equal "
                                  + std::to_string(getSignalLength()) + "\n");
    }
    if (z == nullptr){throw std::invalid_argument("z is NULL\n");}
    if (pUp == nullptr){throw std::invalid_argument("pUp is NULL\n");}
    if (pDown == nullptr){throw std::invalid_argument("pDown is NULL\n");}
    if (pUnknown == nullptr){throw std::invalid_argument("pUnkown is NULL\n");}
    // Default a result
    *pUp = 0;
    *pDown = 0;
    *pUnknown = 1;
    auto X = torch::zeros({1, 1, 400},
                          torch::TensorOptions().dtype(torch::kFloat32)
                          .requires_grad(false));
    float *signalPtr = X.data_ptr<float> ();
    auto lAlive = rescaleAndCopy(pImpl->mSignalLength, z, signalPtr);
    if (!lAlive){return;} // Dead trace nothing more to do
    auto signalGPU = X.to(torch::kCUDA);
    auto pHost = pImpl->mNetwork.forward(signalGPU).to(torch::kCPU);
    float *pPtr = pHost.data_ptr<float> ();
    *pUp = pPtr[0];
    *pDown = pPtr[1];
    *pUnknown = pPtr[2];
}

///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template class UUSS::FirstMotion::FMNet::Model<UUSS::Device::GPU>;
template class UUSS::FirstMotion::FMNet::Model<UUSS::Device::CPU>;
