#include <string>
#include <torch/torch.h>
#include "uuss/oneComponentPicker/zcnn/model.hpp"
#include "private/loadHDF5Weights.hpp"

using namespace UUSS::OneComponentPicker::ZCNN;

#define SIGNAL_LENGTH 400
#define SAMPLING_PERIOD 0.01

namespace
{
/// @brief Computes the absolute max value of an array.
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
/// @brief Performs a min/max normalization and copies.
/// @param[in] npts     The number of points
/// @param[in] z        The vertical trace.  This has dimension [npts].
/// @param[in] n        The north trace.  This has dimension [npts].
/// @param[in] e        The east trae.  This has dimension [npts].
/// @param[out] tensor  The rescaled data for the neural network.
///                     This has dimension [3 x npts] and is stored
///                     row major format.
/// @retval True indicates that this is a live trace.
/// @retval False indicates that this is a dead trace.
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

struct ZCNNPNetwork : torch::nn::Module
{
    // C'tor
    ZCNNPNetwork() :
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
        fcn3(torch::nn::LinearOptions({512, 1}).bias(true))
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
        auto x = conv1->forward(xIn);
        x = torch::relu(x);
        x = batch1->forward(x);
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
        x = torch::hardtanh(x, minVal, maxVal);
        return x;
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
        loader.closeGroup();

        loader.openGroup("/model_bias");
        std::vector<double> biasV;
        std::vector<hsize_t> dims;
        loader.readDataSet("bias", &dims, &biasV);
        bias = biasV.at(0);
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
    const double minVal =-0.75;
    const double maxVal = 0.75;
    double bias = 0;
};

/// Structure with implementation
template<UUSS::Device E>
class Model<E>::ZCNNImpl
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
    ZCNNPNetwork mNetwork;
    //double mSamplingPeriod = 0.01;
    double mBias = 0;
    //int mSignalLength = 400;
    bool mUseGPU = false;
    bool mOnGPU = false;
    bool mHaveCoefficients = false;
    bool mHaveGPU = torch::cuda::is_available();
};

/// C'tor
template<>
Model<UUSS::Device::CPU>::Model() :
    pImpl(std::make_unique<ZCNNImpl> ())
{
    pImpl->mNetwork.eval();
    pImpl->mBias = pImpl->mNetwork.bias;
}

/// C'tor
template<>
Model<UUSS::Device::GPU>::Model () :
    pImpl(std::make_unique<ZCNNImpl> ())
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
    pImpl->mBias = pImpl->mNetwork.bias;
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
int Model<E>::getSignalLength() noexcept
{
    return SIGNAL_LENGTH; //pImpl->mSignalLength;
}

/// Returns the sampling period
template<UUSS::Device E>
double Model<E>::getSamplingPeriod() noexcept
{
    return SAMPLING_PERIOD; //pImpl->mSamplingPeriod;
}

/// Are the model coefficients set?
template<UUSS::Device E>
bool Model<E>::haveModelCoefficients() const noexcept
{
    return pImpl->mHaveCoefficients;
}

/// Predicts a pick time
template<UUSS::Device E>
float Model<E>::predict(int nSamples, const float z[]) const
{
    std::array<float, 1> pickTime;
    constexpr int nSignals = 1;
    constexpr int batchSize = 1;
    auto pickTimePtr = pickTime.data();
    predict(nSignals, nSamples, z, &pickTimePtr, batchSize);
    return pickTime[0];
}

template<UUSS::Device E>
double Model<E>::predict(const int nSamples, const double z[]) const
{
    std::array<double, 1> pickTime;
    constexpr int nSignals = 1;
    constexpr int batchSize = 1;
    auto pickTimePtr = pickTime.data();
    predict(nSignals, nSamples, z, &pickTimePtr, batchSize);
    return pickTime[0];
}

/// Compute the pick times relative to the trace start
template<UUSS::Device E>
void Model<E>::predict(
    const int nSignals, const int nSamplesInSignal, const double z[],
    double *pickTimesIn[], const int batchSize) const
{
    // Preliminary checks
    int signalLength = getSignalLength();
    if (nSignals < 1){throw std::invalid_argument("No signals");}
    if (batchSize < 1)
    {
        throw std::invalid_argument("Batch size must be positive");
    }
    if (nSamplesInSignal != signalLength)
    {
        throw std::invalid_argument("nSamplesInSignal = "
                                  + std::to_string(nSamplesInSignal)
                                  + " must equal "
                                  + std::to_string(signalLength));
    }
    if (z == nullptr){throw std::invalid_argument("z is NULL");}
    auto pickTimes = *pickTimesIn;
    if (pickTimes == nullptr){throw std::invalid_argument("pickTimes is NULL");}
    // Copy signal to float vector
    std::vector<float> zWork(nSignals*signalLength, 0);
    std::copy(z, z + nSignals*signalLength, zWork.data());
    std::vector<float> pickWork(nSignals, 0);
    auto pPtr = pickWork.data(); 
    predict(nSignals, nSamplesInSignal, zWork.data(), &pPtr, batchSize);
    // Copy result back
    std::copy(pPtr, pPtr + nSignals, pickTimes);
}
    
/// Compute the pick times relative to the trace start
template<>
void Model<UUSS::Device::CPU>::predict(
    const int nSignals, const int nSamplesInSignal, const float z[],
    float *pickTimesIn[], const int batchSize) const
{
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients");
    }
    if (nSignals < 1){throw std::invalid_argument("No signals");}
    if (batchSize < 1)
    {
        throw std::invalid_argument("Batch size must be positive");
    }
    int signalLength = getSignalLength();
    if (nSamplesInSignal != signalLength)
    {
        throw std::invalid_argument("nSamplesInSignal = "
                                  + std::to_string(nSamplesInSignal)
                                  + " must equal "
                                  + std::to_string(signalLength));
    }
    if (z == nullptr){throw std::invalid_argument("z is NULL");}
    auto pickTimes = *pickTimesIn;
    if (pickTimes == nullptr){throw std::invalid_argument("pickTimes is NULL");}
    // Initialize result
    float defaultPick = static_cast<float> (getSamplingPeriod()/2*signalLength);
    std::fill(pickTimes, pickTimes + nSignals, defaultPick);
    // Allocate space
    int batchUse = std::min(nSignals, batchSize);
    auto X = torch::zeros({batchUse, 1, signalLength},
                          torch::TensorOptions().dtype(torch::kFloat32)
                          .requires_grad(false));
    std::vector<bool> lAlive(batchUse, true);
    // Loop on batches
    //float minPick = 0.;
    //float maxPick = getSamplingPeriod()*(signalLength - 1);
    float minPick =-getSamplingPeriod()*(signalLength - 1)/2;
    float maxPick =+getSamplingPeriod()*(signalLength - 1)/2;
    for (int is0 = 0; is0 < nSignals; is0 = is0 + batchUse)
    {
        // Extract signal, normalize, and copy
        int js0 = is0;
        int js1 = std::min(nSignals, js0 + batchUse);
        for (int js = js0; js < js1; ++js)
        {
            int iSrc = js*signalLength;
            int iDst = (js - js0)*signalLength;
            float *signalPtr = X.data_ptr<float> () + iDst;
            lAlive[js-js0] = rescaleAndCopy(signalLength, z + iSrc, signalPtr);
        }
        // Predict
        auto p = pImpl->mNetwork.forward(X);
        float *pPtr = p.data_ptr<float> ();
        // Copy picks to output arrays
        for (int js = js0; js < js1; ++js)
        {
            if (lAlive[js - js0])
            {
                pickTimes[js] = std::min(std::max(minPick, pPtr[js - js0]),
                                         maxPick);
            }
        }
    }
}

/// Compute the probability of up/down/unkonwn on a gpu
template<>
void Model<UUSS::Device::GPU>::predict(
    const int nSignals, const int nSamplesInSignal, const float z[],
    float *pickTimesIn[], const int batchSize) const
{
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients");
    }
    if (nSignals < 1){throw std::invalid_argument("No signals");}
    if (batchSize < 1)
    {
        throw std::invalid_argument("Batch size must be positive");
    }
    int signalLength = getSignalLength();
    if (nSamplesInSignal != signalLength)
    {
        throw std::invalid_argument("nSamplesInSignal = "
                                  + std::to_string(nSamplesInSignal)
                                  + " must equal "
                                  + std::to_string(signalLength));
    }
    if (z == nullptr){throw std::invalid_argument("z is NULL");}
    auto pickTimes = *pickTimesIn;
    if (pickTimes == nullptr){throw std::invalid_argument("pickTimes is NULL");}
    float defaultPick = static_cast<float> (getSamplingPeriod()/2*signalLength);
    std::fill(pickTimes, pickTimes + nSignals, defaultPick);
    int batchUse = std::min(nSignals, batchSize);
    auto X = torch::zeros({batchUse, 1, signalLength},
                          torch::TensorOptions().dtype(torch::kFloat32)
                          .requires_grad(false));
    std::vector<bool> lAlive(batchUse, true);
    //float minPick = 0.;
    //float maxPick = getSamplingPeriod()*(signalLength - 1);
    float minPick =-getSamplingPeriod()*(signalLength - 1)/2;
    float maxPick =+getSamplingPeriod()*(signalLength - 1)/2;
    for (int is0 = 0; is0 < nSignals; is0 = is0 + batchUse)
    {
        int js0 = is0;
        int js1 = std::min(nSignals, js0 + batchUse);
        for (int js = js0; js < js1; ++js)
        {
            int iSrc = js*signalLength;
            int iDst = (js - js0)*signalLength;
            float *signalPtr = X.data_ptr<float> () + iDst;
            lAlive[js-js0] = rescaleAndCopy(signalLength, z + iSrc, signalPtr);
        }
        auto signalGPU = X.to(torch::kCUDA);
        auto pHost = pImpl->mNetwork.forward(signalGPU).to(torch::kCPU);
        float *pPtr = pHost.data_ptr<float> ();
        for (int js = js0; js < js1; ++js)
        {
            if (lAlive[js - js0])
            {
                pickTimes[js] = std::min(std::max(minPick, pPtr[js - js0]),
                                         maxPick);
            }
        }
    }
}


///--------------------------------------------------------------------------///
///                              Template Instantiation                      ///
///--------------------------------------------------------------------------///
template class UUSS::OneComponentPicker::ZCNN::Model<UUSS::Device::GPU>;
template class UUSS::OneComponentPicker::ZCNN::Model<UUSS::Device::CPU>;
