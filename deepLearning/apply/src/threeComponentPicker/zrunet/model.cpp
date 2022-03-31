#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional>
#include <iostream>
#if __has_include(<pstl/algorithm>)
   #include <pstl/execution>
   #include <pstl/algorithm>
   #define USE_PSTL 1
#elif __has_include(<experimental/execution>)
   #include <experimental/execution>
   #include <experimental/algorithm>
   #define USE_PSTL 1
#endif
#include <torch/torch.h>
#include <H5Cpp.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <rtseis/utilities/normalization/zscore.hpp>
#include "private/loadHDF5Weights.hpp"
#include "uuss/threeComponentPicker/zrunet/model.hpp"

using namespace UUSS::ThreeComponentPicker::ZRUNet;

namespace
{

template<class T>
void copy(const int n, const T *x, T *y)
{
#ifdef USE_PSTL
    std::copy(std::execution::unseq, x, x+n, y);
#else
    std::copy(x, x + n, y);
#endif
}

/// @brief Maps probabilities to a binary class.
template<class T>
void convertProbabilitiesToClass(const int n,
                                 const float proba[],
                                 T *classIn[],
                                 const float tol = 0.5) 
{
    const T zero = 0;
    const T one = 1;
    T *c = *classIn;
    #pragma omp simd
    for (int i=0; i<n; ++i)
    {
        c[i] = zero;
        if (proba[i] > 0.5){c[i] = one;}
    }
}

/// @brief Computes the accuracy in the predicions.
template<class T>
T computeAccuracy(const int n,
                  const T *yObs,
                  const T *yEst)
{
    T accuracy = 0;
    #pragma omp simd
    for (int i=0; i<n; ++i)
    {
        if (yObs[i] == yEst[i]){accuracy = accuracy + 1;}
    }
    accuracy = accuracy/static_cast<T> (n);
    return accuracy;
}

struct ConfusionMatrix
{
    uint64_t nTruePositives = 0;
    uint64_t nTrueNegatives = 0;
    uint64_t nFalsePositives = 0;
    uint64_t nFalseNegatives = 0; 
};

template<class T>
struct ConfusionMatrix computeConfusionMatrix(const int n,
                                              const T *yObs,
                                              const T *yEst)
{
    const T twoTol  = 2 - 100*std::numeric_limits<T>::epsilon();
    const T zeroTol = 100*std::numeric_limits<T>::epsilon();
    uint64_t ntp = 0;
    uint64_t nfp = 0;
    uint64_t ntn = 0;
    uint64_t nfn = 0;
    #pragma omp simd reduction(+:ntp, nfp)
    for (int i=0; i<n; ++i)
    {
        auto sum = yObs[i] + yEst[i];
        // Sum is small - both predict negative (true negative)
        if (sum < zeroTol)
        {
            ntn = ntn + 1; 
        }
        // Sum is one + one - both predict positive (true positive)
        else if (sum > twoTol)
        {
            ntp = ntp + 1;
        }
        else
        {
            // Observe negative (predict positive)
            if (yObs[i] < zeroTol)
            {
                nfp = nfp + 1; 
            }
            else
            {
                nfn = nfn + 1;
            }            
        }
    }
    struct ConfusionMatrix cm;
    cm.nTruePositives = ntp;
    cm.nTrueNegatives = ntn;
    cm.nFalseNegatives = nfn;
    cm.nFalsePositives = nfp;
    return cm;
}

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

template<class T> T getRange(const int npts, const T *__restrict__ v)
{
#ifdef USE_PSTL
    auto [vMin, vMax] = std::minmax_element(std::execution::unseq, v, v+npts);
#else
    auto [vMin, vMax] = std::minmax_element(v, v+npts);
#endif
    T range = vMax - vMin;
    return range;
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
                    const T *__restrict__ n, 
                    const T *__restrict__ e,
                    T *__restrict__ tensor)
{
    constexpr T zero = 0;
    auto zMax = getMaxAbs(npts, z);
    auto nMax = getMaxAbs(npts, n);
    auto eMax = getMaxAbs(npts, e);
    auto maxVal = std::max(zMax, std::max(nMax, eMax));
    if (maxVal != 0)
    {
        auto xnorm = static_cast<T> (1.0/maxVal);
#ifdef USE_PSTL
        std::transform(std::execution::unseq,
                       n, n + npts, tensor,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
        std::transform(std::execution::unseq,
                       e, e + npts, tensor + npts,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
        std::transform(std::execution::unseq,
                       z, z + npts, tensor + 2*npts,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
#else
        std::transform(n, n + npts, tensor,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
        std::transform(e, e + npts, tensor + npts,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
        std::transform(z, z + npts, tensor + 2*npts,
                       std::bind(std::multiplies<T>(),
                       std::placeholders::_1, xnorm));
#endif
        return true;
    }
    else
    {
#ifdef USE_PSTL
        std::fill(std::execution::unseq, tensor, tensor + 3*npts, zero);
#else
        std::fill(tensor, tensor + 3*npts, zero);
#endif
        return false;
    } 
}

/// @brief Performs a Z-Score normalization to each channel and copies.
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
bool standardizeAndCopy(const int npts,
                        const T *__restrict__ z,
                        const T *__restrict__ n, 
                        const T *__restrict__ e,
                        T *__restrict__ tensor)
{
    constexpr T zero = 0;
    constexpr T tol = std::numeric_limits<T>::epsilon();
    auto zRange = getRange(npts, z);
    auto nRange = getRange(npts, n);
    auto eRange = getRange(npts, e);
    bool notDead = false;
    if (npts < 2){throw std::invalid_argument("At least 2 points needed");}
    RTSeis::Utilities::Normalization::ZScore zScore;
    // Standardize N.
    // N.B. There's a check in RTSeis to avoid a division by zero.  However,
    // that's only possible to hit if all the samples in the signal are
    // identical.  The range condition ensures this is not the case.
    if (std::abs(nRange) > tol)
    {
        zScore.initialize(npts, n);
        auto tPtr = tensor;
        zScore.apply(npts, n, &tPtr);
        notDead = true;
    }
    else
    {
        std::fill(tensor + npts, tensor + npts, zero);
    }
    // Standardize E
    if (std::abs(eRange) > tol)
    {
        zScore.initialize(npts, e);
        auto tPtr = tensor + npts;
        zScore.apply(npts, e, &tPtr);
        notDead = true;
    }
    else
    {
        std::fill(tensor + npts, tensor + 2*npts, zero);
    }
    // Standardize Z
    if (std::abs(zRange) > tol)
    {
        zScore.initialize(npts, z);
        auto tPtr = tensor + 2*npts;
        zScore.apply(npts, z, &tPtr);
        notDead = true;
    }
    else
    {
        std::fill(tensor + 2*npts, tensor + 3*npts, zero);
    }
    return notDead;
}

/// @brief Defines the UNet network architecture.
/// @copyright Ben Baker distributed under the MIT license.
struct UNet : torch::nn::Module
{
    /// Constructor
    UNet(const int nInputChannels = 3,
         const int nOutputChannels = 1) :
        mInputChannels(nInputChannels),
        mOutputChannels(nOutputChannels),
        // 1
        conv11(torch::nn::Conv1dOptions(mInputChannels, 64, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv12(torch::nn::Conv1dOptions(64, 64, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch1(torch::nn::BatchNormOptions(64)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 2
        conv21(torch::nn::Conv1dOptions(64, 128, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv22(torch::nn::Conv1dOptions(128, 128, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch2(torch::nn::BatchNormOptions(128)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 3
        conv31(torch::nn::Conv1dOptions(128, 256, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv32(torch::nn::Conv1dOptions(256, 256, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch3(torch::nn::BatchNormOptions(256)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 4
        conv41(torch::nn::Conv1dOptions(256, 512, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv42(torch::nn::Conv1dOptions(512, 512, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch4(torch::nn::BatchNormOptions(512)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 5
        conv51(torch::nn::Conv1dOptions(512, 1024, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv52(torch::nn::Conv1dOptions(1024, 1024, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch5(torch::nn::BatchNormOptions(1024)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 6 -> begin expansion
        uconv6(torch::nn::ConvTranspose1dOptions(1024, 512, mUpsampleKernelSize)
              .stride(mUpsampleStride).bias(true)),
        conv61(torch::nn::Conv1dOptions(1024, 512, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv62(torch::nn::Conv1dOptions(512, 512, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch6(torch::nn::BatchNormOptions(512)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 7
        uconv7(torch::nn::ConvTranspose1dOptions(512, 256, mUpsampleKernelSize)
              .stride(mUpsampleStride).bias(true)),
        conv71(torch::nn::Conv1dOptions(512, 256, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv72(torch::nn::Conv1dOptions(256, 256, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch7(torch::nn::BatchNormOptions(256)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 8
        uconv8(torch::nn::ConvTranspose1dOptions(256, 128, mUpsampleKernelSize)
              .stride(mUpsampleStride).bias(true)),
        conv81(torch::nn::Conv1dOptions(256, 128, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv82(torch::nn::Conv1dOptions(128, 128, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch8(torch::nn::BatchNormOptions(128)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        // 9
        uconv9(torch::nn::ConvTranspose1dOptions(128, 64, mUpsampleKernelSize)
              .stride(mUpsampleStride).bias(true)),
        conv91(torch::nn::Conv1dOptions(128, 64, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        conv92(torch::nn::Conv1dOptions(64, 64, mKernelSize)
           .stride(1).padding(mPadding).bias(true).dilation(1)),
        batch9(torch::nn::BatchNormOptions(64)
           .eps(1.e-5).momentum(0.1).affine(true).track_running_stats(true)),
        conv93(torch::nn::Conv1dOptions(64, mOutputChannels, 1) // Not padding
           .stride(1).padding(0).bias(true).dilation(1))
    {
        register_module("conv1d_1_1", conv11);
        register_module("conv1d_1_2", conv12);
        register_module("batch_normalization_1", batch1);

        register_module("conv1d_2_1", conv21);
        register_module("conv1d_2_2", conv22);
        register_module("batch_normalization_2", batch2);

        register_module("conv1d_3_1", conv31);
        register_module("conv1d_3_2", conv32);
        register_module("batch_normalization_3", batch3);

        register_module("conv1d_4_1", conv41);
        register_module("conv1d_4_2", conv42);
        register_module("batch_normalization_4", batch4);

        register_module("conv1d_5_1", conv51);
        register_module("conv1d_5_2", conv52);
        register_module("batch_normalization_5", batch5);

        register_module("convTranspose1d_6_1", uconv6);
        register_module("conv1d_6_1", conv61);
        register_module("conv1d_6_2", conv62);
        register_module("batch_normalization_6", batch6);

        register_module("convTranspose1d_7_1", uconv7);
        register_module("conv1d_7_1", conv71);
        register_module("conv1d_7_2", conv72);
        register_module("batch_normalization_7", batch7);

        register_module("convTranspose1d_8_1", uconv8);
        register_module("conv1d_8_1", conv81);
        register_module("conv1d_8_2", conv82);
        register_module("batch_normalization_8", batch8);

        register_module("convTranspose1d_9_1", uconv9);
        register_module("conv1d_9_1", conv91);
        register_module("conv1d_9_2", conv92);
        register_module("batch_normalization_9", batch9);

        register_module("conv1d_9_3", conv93);
    }

    /// Forward
    torch::Tensor forward(const torch::Tensor &xIn,
                          const bool applySigmoid = true)
    {
        // [1 x 240]
        auto x = conv11->forward(xIn); // [1 x 64 x 240]
        x = torch::relu(x);
        x = conv12->forward(x);        // [1 x 64 x 240]
        auto x1d = torch::relu(x);
        x1d = batch1->forward(x1d);
        x = torch::max_pool1d(x1d, mMaxPoolSize, mPoolStride); // [1 x 64 x 120]
        //printf("x1d.size: %u, %u, %u\n", x1d.size(0), x1d.size(1), x1d.size(2));

        x = conv21->forward(x);        // [1 x 128 x 120]
        x = torch::relu(x);
        x = conv22->forward(x);        // [1 x 128 x 120]
        auto x2d = torch::relu(x);
        x2d = batch2->forward(x2d);
        x = torch::max_pool1d(x2d, mMaxPoolSize, mPoolStride); // [1 x 128 x 60]
        //printf("x2d.size: %u, %u, %u\n", x2d.size(0), x2d.size(1), x2d.size(2));

        x = conv31->forward(x);        // [1 x 256 x 60]
        x = torch::relu(x);
        x = conv32->forward(x);        // [1 x 256 x 60]
        auto x3d = torch::relu(x);
        x3d = batch3->forward(x3d);
        x = torch::max_pool1d(x3d, mMaxPoolSize, mPoolStride); // [1 x 256 x 30]
        //printf("x3d.size: %u, %u, %u\n", x3d.size(0), x3d.size(1), x3d.size(2));

        x = conv41->forward(x);        // [1 x 512 x 30]
        x = torch::relu(x);
        x = conv42->forward(x);        // [1 x 512 x 30]
        auto x4d = torch::relu(x);
        x4d = batch4->forward(x4d);
        x = torch::max_pool1d(x4d, mMaxPoolSize, mPoolStride); // [1 x 512 x 15]
        //printf("x4d.size: %u, %u, %u\n", x4d.size(0), x4d.size(1), x4d.size(2));

        x = conv51->forward(x);        // [1 x 1024 x 15]
        x = torch::relu(x);
        x = conv52->forward(x);        // [1 x 1024 x 15]
        auto x5d = torch::relu(x);
        x5d = batch5->forward(x5d);

        //printf("x5d.size: %u, %u, %u\n", x5d.size(0), x5d.size(1), x5d.size(2));
        // Begin upsampling
        x = uconv6->forward(x5d);      // [1 x 512 x 30] (upsample 2x)
        x = torch::cat({x4d, x}, 1);   // [1 x 512 x 30]
        x = conv61->forward(x);        // [1 x 512 x 30]
        x = torch::relu(x);
        x = conv62->forward(x);        // [1 x 512 x 30]
        x = torch::relu(x);
        x = batch6->forward(x);

        auto x7u = uconv7->forward(x); // [1 x 256 x 60] (upsample 2x)
        x = torch::cat({x3d, x7u}, 1);
        x = conv71->forward(x);        // [1 x 256 x 60]
        x = torch::relu(x);
        x = conv72->forward(x);        // [1 x 256 x 60] 
        x = torch::relu(x);
        x = batch7->forward(x);

        auto x8u = uconv8->forward(x); // [1 x 128 x 120] (upsample 2x)
        x = torch::cat({x2d, x8u}, 1); 
        x = conv81->forward(x);        // [1 x 128 x 120]
        x = torch::relu(x);
        x = conv82->forward(x);        // [1 x 128 x 120]
        x = torch::relu(x);
        x = batch8->forward(x);

        auto x9u = uconv9->forward(x); // [1 x 64 x 240] (upsample 2x)
        x = torch::cat({x1d, x9u}, 1);
        x = conv91->forward(x);        // [1 x 64 x 240]
        x = torch::relu(x);
        x = conv92->forward(x);        // [1 x 64 x 240]
        x = torch::relu(x);
        x = batch9->forward(x);

        x = conv93->forward(x);        // [1 x 240]

        if (applySigmoid){x = torch::sigmoid(x);}     // [1 x 240]
        return x;
    }
    /// Loads the weights from file
    void loadWeightsFromHDF5(const std::string &fileName,
                             const bool gpu = false,
                             const bool verbose = false,
                             const int device =-1)
    {
        HDF5Loader loader;
        loader.openFile(fileName);
        loader.openGroup("/model_weights/sequential_1"); 
        
        readWeightsAndBiasFromHDF5(loader, "conv1d_1_1",
                                   conv11, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_1_2",
                                   conv12, gpu, verbose, device); 
        readBatchNormalizationWeightsFromHDF5(loader, "bn_1", batch1,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "conv1d_2_1",
                                   conv21, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_2_2",
                                   conv22, gpu, verbose, device); 
        readBatchNormalizationWeightsFromHDF5(loader, "bn_2", batch2,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "conv1d_3_1",
                                   conv31, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_3_2",
                                   conv32, gpu, verbose, device); 
        readBatchNormalizationWeightsFromHDF5(loader, "bn_3", batch3,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "conv1d_4_1",
                                   conv41, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_4_2",
                                   conv42, gpu, verbose, device);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_4", batch4,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "conv1d_5_1",
                                   conv51, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_5_2",
                                   conv52, gpu, verbose, device);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_5", batch5,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "convTranspose1d_6_1",
                                   uconv6, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_6_1",
                                   conv61, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_6_2",
                                   conv62, gpu, verbose, device);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_6", batch6,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "convTranspose1d_7_1",
                                   uconv7, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_7_1",
                                   conv71, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_7_2",
                                   conv72, gpu, verbose, device);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_7", batch7,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "convTranspose1d_8_1",
                                   uconv8, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_8_1",
                                   conv81, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_8_2",
                                   conv82, gpu, verbose, device);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_8", batch8,
                                              gpu, verbose, device);

        readWeightsAndBiasFromHDF5(loader, "convTranspose1d_9_1",
                                   uconv9, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_9_1",
                                   conv91, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_9_2",
                                   conv92, gpu, verbose, device);
        readWeightsAndBiasFromHDF5(loader, "conv1d_9_3",
                                   conv93, gpu, verbose, device);
        readBatchNormalizationWeightsFromHDF5(loader, "bn_9", batch9,
                                              gpu, verbose, device);
    }
    /// Write the weights to a file
    /*
    void writeWeightsToHDF5(const std::string &fileName,
                            const bool gpu = false,
                            const bool verbose = false)
    {
        H5::Exception::dontPrint();
        H5::H5File h5File(fileName, H5F_ACC_TRUNC);
        // Create groups
        h5File.createGroup("/model_weights");
        auto group = h5File.createGroup("/model_weights/sequential_1");
        for (const auto &pair : named_parameters())
        {
            dumpWeightsToHDF5(group, pair.key(), pair.value(), gpu, verbose);
        }
        h5File.close();
    }
    */
///private:
    std::vector<std::string> h5WeightNames;
    std::vector<std::string> h5BiasNames;

    const int64_t mKernelSize = 3;
    const int64_t mPadding = 1; //= kernelaSize/Stride = 3/2 
    const int64_t mUpsampleStride = 2;
    const int64_t mUpsampleKernelSize = 2;
    const int64_t mMaxPoolSize = 2;
    const int64_t mPoolStride = 2;
    int64_t mInputChannels = 3;
    int64_t mOutputChannels = 1;

    torch::nn::Conv1d conv11{nullptr};
    torch::nn::Conv1d conv12{nullptr};
    torch::nn::BatchNorm1d batch1{nullptr};

    torch::nn::Conv1d conv21{nullptr};
    torch::nn::Conv1d conv22{nullptr};
    torch::nn::BatchNorm1d batch2{nullptr};

    torch::nn::Conv1d conv31{nullptr};
    torch::nn::Conv1d conv32{nullptr};
    torch::nn::BatchNorm1d batch3{nullptr};

    torch::nn::Conv1d conv41{nullptr};
    torch::nn::Conv1d conv42{nullptr};
    torch::nn::BatchNorm1d batch4{nullptr};

    torch::nn::Conv1d conv51{nullptr};
    torch::nn::Conv1d conv52{nullptr};
    torch::nn::BatchNorm1d batch5{nullptr};

    torch::nn::ConvTranspose1d uconv6{nullptr};
    torch::nn::Conv1d conv61{nullptr};
    torch::nn::Conv1d conv62{nullptr};
    torch::nn::BatchNorm1d batch6{nullptr};

    torch::nn::ConvTranspose1d uconv7{nullptr};
    torch::nn::Conv1d conv71{nullptr};
    torch::nn::Conv1d conv72{nullptr};
    torch::nn::BatchNorm1d batch7{nullptr};

    torch::nn::ConvTranspose1d uconv8{nullptr};
    torch::nn::Conv1d conv81{nullptr};
    torch::nn::Conv1d conv82{nullptr};
    torch::nn::BatchNorm1d batch8{nullptr};

    torch::nn::ConvTranspose1d uconv9{nullptr};
    torch::nn::Conv1d conv91{nullptr};
    torch::nn::Conv1d conv92{nullptr};
    torch::nn::BatchNorm1d batch9{nullptr};

    torch::nn::Conv1d conv93{nullptr};
};

}
//----------------------------------------------------------------------------//
//                           End anonamous namespace                          //
//----------------------------------------------------------------------------//

/// @brief Implementation.
template<UUSS::Device E>
class Model<E>::ModelImpl
{
public:
    ModelImpl() :
        mHaveGPU(torch::cuda::is_available())
    {
    }
    void toGPU()
    {
        if (!mOnGPU && mUseGPU && mHaveGPU)
        {
            for (const auto deviceID : mDeviceIDs)
            {
                auto dev16 = static_cast<int8_t> (deviceID);
                mNetworks[deviceID].to({torch::kCUDA, dev16});
            }
            mOnGPU = true;
        }
    }
    //UNet mNetwork; 
    std::vector<UNet> mNetworks;
    /// Classifies to group 0 when less than and group 1 when greater than tol
    double mPredictTol = 0.5;
    const int mMinimumSeismogramLength = 16;
    std::vector<int> mDeviceIDs;
    int mDeviceIDUse = 0;
    bool mUseGPU = false;
    bool mOnGPU = false; /// Check if this is set on GPU yet
    bool mHaveCoefficients = false;
    bool mHaveGPU = torch::cuda::is_available();
};

/// Constructor for CPU
template<>
Model<UUSS::Device::CPU>::Model() :
    pImpl(std::make_unique<ModelImpl> ())
{
    pImpl->mNetworks.resize(1);
    pImpl->mDeviceIDs.resize(1);
    pImpl->mDeviceIDs[0] = 0;
    pImpl->mNetworks[0].eval();
}

/// Constructor for GPU
template<>
Model<UUSS::Device::GPU>::Model() :
    pImpl(std::make_unique<ModelImpl> ())
{
    if (!pImpl->mHaveGPU)
    {
        throw std::runtime_error("CUDA not available"); // - using CPU" << std::endl;
        pImpl->mNetworks.resize(1);
        pImpl->mDeviceIDs.resize(1);
        pImpl->mDeviceIDs[0] = 0;
        pImpl->mNetworks[0].eval();
    }
    else
    {
        pImpl->mUseGPU = true;
        auto nGPUs = static_cast<int> (torch::cuda::device_count());
        for (int gpuID = 0; gpuID < nGPUs; ++gpuID)
        {
            pImpl->mDeviceIDs.push_back(gpuID);
        }
/*
        for (int gpuID = 0; gpuID < nGPUs; ++gpuID)
        {
            cudaSetDevice(gpuID);
            int deviceID = 0;
            cudaGetDevice(&deviceID);
            std::cout << "Detected device ID: " << deviceID << std::endl; 
            pImpl->mDeviceIDs.push_back(deviceID);
        }
*/
        std::cout << "Number of devices found: "
                  << pImpl->mDeviceIDs.size() << std::endl;
        pImpl->mNetworks.resize(nGPUs);
        for (int gpuID = 0;  gpuID < nGPUs; ++gpuID)
        {
            pImpl->mNetworks[gpuID].eval();
        }
    }
    pImpl->toGPU();
    //pImpl->mNetwork.eval();
}

/// Destructor
template<UUSS::Device E>
Model<E>::~Model() = default;

/// Move c'tor
template<UUSS::Device E>
Model<E>::Model(Model &&model) noexcept
{
    *this = std::move(model);
}

/// Move assignment
template<UUSS::Device E>
Model<E>& Model<E>::operator=(Model<E> &&model) noexcept
{
    if (&model == this){return *this;}
    pImpl = std::move(model.pImpl);
    return *this;
}

/// Load from weights
template<UUSS::Device E>
void Model<E>::loadWeightsFromHDF5(const std::string &fileName,
                                   const bool verbose)
{
    if (!pImpl->mUseGPU)
    {
        pImpl->mNetworks[0].loadWeightsFromHDF5(fileName, false, verbose);
        pImpl->mNetworks[0].eval();
    }
    else
    {
        for (const auto &dev : pImpl->mDeviceIDs)
        {
            if (dev == pImpl->mDeviceIDUse)
            {
                std::cout << "Loading weights for device: " << dev << std::endl;
                pImpl->mNetworks[dev].loadWeightsFromHDF5(fileName, true,
                                                          verbose, dev); 
                pImpl->mNetworks[dev].eval();
            }
        }
        //pImpl->toGPU();
    }
    //pImpl->mNetwork.eval();
    pImpl->mHaveCoefficients = true;
}

/// Classifies the phase at all all points in the time series
template<UUSS::Device E>
void Model<E>::predict(const int nSamples,
                       const float vertical[],
                       const float north[],
                       const float east[],
                       int *clssIn[]) const
{
    const float tol = static_cast<float> (pImpl->mPredictTol);
    if (*clssIn == nullptr){throw std::invalid_argument("clss is NULL\n");}
    std::vector<float> proba(nSamples);
    auto probaPtr = proba.data();
    predictProbability(nSamples, vertical, north, east, &probaPtr);
    convertProbabilitiesToClass(nSamples, probaPtr, clssIn, tol);
}

/// Double precision interface
template<UUSS::Device E>
void Model<E>::predictProbability(int nSamples,
                                  const double vertical[],
                                  const double north[],
                                  const double east[],
                                  double *probaIn[]) const
{
    if (!isValidSeismogramLength(nSamples) != 0)
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must be a multiple of "
                                  + std::to_string(16));
    }
    double *proba = *probaIn;
    if (vertical == nullptr || north == nullptr ||
        east == nullptr || proba == nullptr)
    {
        if (vertical == nullptr)
        {
            throw std::invalid_argument("vertical is NULL");
        }
        if (north == nullptr){throw std::invalid_argument("north is NULL");}
        if (east == nullptr){throw std::invalid_argument("east is NULL");}
        throw std::invalid_argument("proba is NULL");
    } 
    std::vector<float> v4(nSamples), n4(nSamples), e4(nSamples), p4(nSamples);
    std::copy(vertical, vertical+nSamples, v4.data());
    std::copy(north,    north+nSamples,    n4.data());
    std::copy(east,     east+nSamples,     e4.data());
    float *p4Ptr = p4.data();
    predictProbability(nSamples, v4.data(), n4.data(), e4.data(), &p4Ptr); 
    std::copy(p4.begin(), p4.end(), proba);
}

/// Predicts the probability of a phase at all points in the time series
template<>
void Model<UUSS::Device::CPU>::predictProbability(
    const int nSamples,
    const float vertical[], const float north[], const float east[],
    float *probaIn[]) const
{
    constexpr bool applySigmoid = true;
    // Require the model be in evaluation mode
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients");
    }
    //pImpl->mNetworks[0].eval();
    if (!isValidSeismogramLength(nSamples) != 0)
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must be a multiple of "
                                  + std::to_string(16));
    }
    float *proba = *probaIn;
    if (vertical == nullptr || north == nullptr ||
        east == nullptr || proba == nullptr)
    {
        if (vertical == nullptr)
        {
            throw std::invalid_argument("vertical is NULL");
        }
        if (north == nullptr){throw std::invalid_argument("north is NULL");}
        if (east == nullptr){throw std::invalid_argument("east is NULL");}
        throw std::invalid_argument("proba is NULL");
    }
    // Allocate space
    auto X = torch::zeros({1, 3, nSamples},
                          torch::TensorOptions().dtype(torch::kFloat32)
                          .requires_grad(false));
    float *xPtr = X.data_ptr<float> ();
    // Feature rescale
    rescaleAndCopy(nSamples, vertical, north, east, xPtr);
    auto p = pImpl->mNetworks[0].forward(X, applySigmoid);
    float *pPtr = p.data_ptr<float> ();
    copy(nSamples, pPtr, proba);
}

template<>
void Model<UUSS::Device::GPU>::predictProbability(
    const int nSamples,
    const float vertical[], const float north[], const float east[],
    float *probaIn[]) const
{
    constexpr bool applySigmoid = true;
    // Require the model be in evaluation mode
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients");
    }
    //pImpl->mNetworks[0].eval();
    if (!isValidSeismogramLength(nSamples) != 0)
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must be a multiple of "
                                  + std::to_string(16));
    }
    float *proba = *probaIn;
    if (vertical == nullptr || north == nullptr ||
        east == nullptr || proba == nullptr)
    {
        if (vertical == nullptr)
        {
            throw std::invalid_argument("vertical is NULL");
        }
        if (north == nullptr){throw std::invalid_argument("north is NULL");}
        if (east == nullptr){throw std::invalid_argument("east is NULL");}
        throw std::invalid_argument("proba is NULL");
    }
    // Allocate space
    auto X = torch::zeros({1, 3, nSamples},
                          torch::TensorOptions().dtype(torch::kFloat32)
                          .requires_grad(false));
    float *xPtr = X.data_ptr<float> ();
    // Feature rescale
    rescaleAndCopy(nSamples, vertical, north, east, xPtr);
    auto XGPU = X.to({torch::kCUDA, pImpl->mDeviceIDUse}); //pImpl->mDeviceIDs[0]});
    auto p = pImpl->mNetworks[0].forward(XGPU, applySigmoid);
    // Copy the answer
    auto pHost = p.to(torch::kCPU);
    float *pPtr = pHost.data_ptr<float> ();
    copy(nSamples, pPtr, proba);
}

/// Double precision interface
template<UUSS::Device E>
void Model<E>::predictProbability(
    const int nSamples, const int nSamplesInWindow, const int nCenter,
    const double vertical[],  const double north[], const double east[],
    double *probaIn[], const int batchSize) const
{
    if (!isValidSeismogramLength(nSamples) != 0)
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must be a multiple of "
                                  + std::to_string(16));
    }    
    double *proba = *probaIn;
    if (vertical == nullptr || north == nullptr ||
        east == nullptr || proba == nullptr)
    {
        if (vertical == nullptr)
        {
            throw std::invalid_argument("vertical is NULL");
        }
        if (north == nullptr){throw std::invalid_argument("north is NULL");}
        if (east == nullptr){throw std::invalid_argument("east is NULL");}
        throw std::invalid_argument("proba is NULL");
    }    
    std::vector<float> v4(nSamples), n4(nSamples), e4(nSamples), p4(nSamples);
    std::copy(vertical, vertical + nSamples, v4.data());
    std::copy(north,    north + nSamples,    n4.data());
    std::copy(east,     east + nSamples,     e4.data());
    float *p4Ptr = p4.data();
    predictProbability(nSamples, nSamplesInWindow, nCenter,
                       v4.data(), n4.data(), e4.data(), &p4Ptr,
                       batchSize); 
    std::copy(p4.begin(), p4.end(), proba);
}

/// General case for sliding window
template<>
void Model<UUSS::Device::CPU>::predictProbability(
    const int nSamples, const int nSamplesInWindow, const int nCenter,
    const float vertical[],  const float north[], const float east[],
    float *probaIn[], const int batchSize) const
{
    constexpr bool applySigmoid = true;
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients");
    }
    if (nSamplesInWindow > nSamples)
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must exceed nSamplesInWindow = "
                                  + std::to_string(nSamplesInWindow)); 
    }
    if (!isValidSeismogramLength(nSamplesInWindow) != 0)
    {
        throw std::invalid_argument("nSamplesInWindow = "
                                  + std::to_string(nSamplesInWindow)
                                  + " must be a multiple of "
                                  + std::to_string(16));
    }
    if (batchSize < 1)
    {
        throw std::invalid_argument("Batch size must be positive");
    }
    float *proba = *probaIn;
    if (vertical == nullptr || north == nullptr ||
        east == nullptr || proba == nullptr)
    {
        if (vertical == nullptr)
        {
            throw std::invalid_argument("vertical is NULL");
        }
        if (north == nullptr){throw std::invalid_argument("north is NULL");}
        if (east == nullptr){throw std::invalid_argument("east is NULL");}
        throw std::invalid_argument("proba is NULL");
    }
    // Both algorithms need to track the last probability index  accessed
    int lastIndex = 0;
    // This is the easiest case where we slide the full window.
    if (nCenter == 0)
    {
        // Allocate space
        auto X = torch::zeros({batchSize, 3, nSamplesInWindow},
                               torch::TensorOptions().dtype(torch::kFloat32)
                              .requires_grad(false));
        // Feature rescale
        int nUpdate = nSamplesInWindow*batchSize;
        for (int iStart = 0; iStart < nSamples; iStart = iStart + nUpdate)
        {
            // Copy the windowed signals
            int nWindows = 0;
            int nAdvance = nSamplesInWindow;
            for (int batch = 0; batch < batchSize; ++batch)
            {
                int i1 = iStart + batch*nAdvance;
                int i2 = i1 + nSamplesInWindow;
                if (i2 > nSamples){break;}
                int j1 = 3*batch*nSamplesInWindow; 
                float *xPtr = X.data_ptr<float> () + j1;
                rescaleAndCopy(nSamplesInWindow,
                               vertical+i1, north+i1, east+i1, xPtr);
                nWindows = nWindows + 1;
            }
            // Apply and copy back
            int nCopy = nWindows*nSamplesInWindow;
            assert(nCopy + iStart <= nSamples);
            auto p = pImpl->mNetworks[0].forward(X, applySigmoid);
            float *pPtr = p.data_ptr<float> ();
            copy(nCopy, pPtr, proba + iStart);
            lastIndex = iStart + nCopy;
        }
    }
    else
    {
        // The first window is an edgecase.  Fill the first nSamplesInWindow
        // probabilities.  Note, we may overwrite some of these depending on
        // nCenter.
        predictProbability(nSamplesInWindow, vertical, north, east, &proba);
        // Initialize the iteration.  The classifier works by looking around
        // [nHalf-center:nHalf+center].  By virtue of computing the first 
        // edge case we already have [0:nHalf+center] so that's where we start
        // inserting our results into the destination vector.
        int nHalf = nSamplesInWindow/2; // Center window around middle
        int iDst = nHalf + nCenter;
        // Obviously, we need to orient the data source window so that it
        // will center on: [nHalf+center:nHalf+2*center+center].
        // A little algebra revelas the central point for the first window
        // in the iteration is:
        // (nHalf + center + nHalf + 3*center)/2 = nHalf + 2*center.
        // We need to strip off the first nHalf/2 so the first index is
         // nhalf + 2*center - nHalf = 2*center
        int iSrc = 2*nCenter; // Already did nHalf - center
        int nAdvanceWindow = 2*nCenter; // Move forward central window size
        // Allocate space
        auto X = torch::zeros({batchSize, 3, nSamplesInWindow},
                               torch::TensorOptions().dtype(torch::kFloat32)
                              .requires_grad(false));
        // This is a worst-case scenario loop that enclosed break statements
        // will force to terminate early.  while(true) loops are discomforting.
        for (int kloop = 0; kloop < nSamples; ++kloop)
        {
            int nWindows = 0;
            for (int batch=0; batch<batchSize; ++batch)
            {
                int i1 = iSrc + batch*nAdvanceWindow;
                int i2 = i1 + nSamplesInWindow;
                if (i2 > nSamples){break;}
                int j1 = 3*batch*nSamplesInWindow;
                float *xPtr = X.data_ptr<float> () + j1;
                rescaleAndCopy(nSamplesInWindow,
                               vertical+i1, north+i1, east+i1, xPtr);
                nWindows = nWindows + 1;
            }
            auto p = pImpl->mNetworks[0].forward(X, applySigmoid);
            float *pPtr = p.data_ptr<float> ();
            for (int iwin=0; iwin<nWindows; iwin++)
            {
                int jSrc = nHalf - nCenter + iwin*nSamplesInWindow;
                int jDst = iDst + iwin*nAdvanceWindow;
                copy(nAdvanceWindow, pPtr + jSrc, proba + jDst);
            }
            // Update counters
            iSrc = iSrc + nAdvanceWindow*nWindows;
            iDst = iDst + nAdvanceWindow*nWindows;
            lastIndex = iDst;
            if (iSrc + nSamplesInWindow > nSamples){break;}
        }
    }
    // Now fill the lastIndex:nSamples probabilities
    int nRemainder = nSamples - lastIndex;
    if (nRemainder > 0) 
    {
        //printf("In cleanup\n");
        int i1 = nSamples - nSamplesInWindow;
        auto Xlast = torch::zeros({1, 3, nSamplesInWindow},
                                   torch::TensorOptions().dtype(torch::kFloat32)
                                  .requires_grad(false));
        float *xPtr = Xlast.data_ptr<float> ();
        rescaleAndCopy(nSamplesInWindow,
                       vertical+i1, north+i1, east+i1, xPtr);
        int j1 = nSamplesInWindow - nRemainder;
        auto p = pImpl->mNetworks[0].forward(Xlast, applySigmoid);
        float *pPtr = p.data_ptr<float> ();
        copy(nRemainder, pPtr + j1, proba + lastIndex);
    }
}

template<>
void Model<UUSS::Device::GPU>::predictProbability(
    const int nSamples, const int nSamplesInWindow, const int nCenter,
    const float vertical[], const float north[], const float east[],
    float *probaIn[], const int batchSize) const
{
    constexpr bool applySigmoid = true;
    if (!haveModelCoefficients())
    {
        throw std::runtime_error("Model does not have coefficients");
    }
    if (nSamplesInWindow > nSamples)
    {
        throw std::invalid_argument("nSamples = " + std::to_string(nSamples)
                                  + " must exceed nSamplesInWindow = "
                                  + std::to_string(nSamplesInWindow));
    }
    if (!isValidSeismogramLength(nSamplesInWindow) != 0)
    {
        throw std::invalid_argument("nSamplesInWindow = "
                                  + std::to_string(nSamplesInWindow)
                                  + " must be a multiple of "
                                  + std::to_string(16));
    }
    if (batchSize < 1)
    {
        throw std::invalid_argument("Batch size must be positive");
    }
    float *proba = *probaIn;
    if (vertical == nullptr || north == nullptr ||
        east == nullptr || proba == nullptr)
    {
        if (vertical == nullptr)
        {
            throw std::invalid_argument("vertical is NULL");
        }
        if (north == nullptr){throw std::invalid_argument("north is NULL");}
        if (east == nullptr){throw std::invalid_argument("east is NULL");}
        throw std::invalid_argument("proba is NULL");
    }
#if defined(_OPENMP)
    int nOriginalThreads = omp_get_max_threads();
#endif
    //int nDevices = getNumberOfDevices();
    //int nThreadsUse = std::min(nOriginalThreads, nDevices);
#if defined(_OPENMP)
    int nThreadsUse = 1;
    omp_set_num_threads(nThreadsUse);
#endif
    auto deviceIDs = pImpl->mDeviceIDs; 
    int lastIndex = 0;
    UNet *networks = pImpl->mNetworks.data();
    if (nCenter == 0)
    {
        //auto X = torch::zeros({batchSize, 3, nSamplesInWindow},
        //                       torch::TensorOptions().dtype(torch::kFloat32)
        //                      .requires_grad(false));
        int iStart = 0;
        for (int kloop=0; kloop<nSamples; ++kloop) // int iStart=0; iStart < nSamples; iStart = iStart + nUpdate)
        {
            int nCopy = 0;
/*
            #pragma omp parallel \
             shared(batchSize, std::cout, iStart, vertical, north, east) \
             shared(nSamples, nSamplesInWindow, proba, deviceIDs, networks) \
             shared(torch::kFloat32, torch::kCPU, torch::kCUDA) \
             default(none) \
             reduction(+:nCopy)
*/
            {
            int threadID = 0;
#if defined(_OPENMP)
            threadID = omp_get_thread_num();
#endif
            int deviceID = deviceIDs[threadID];
            auto X = torch::zeros({batchSize, 3, nSamplesInWindow},
                                   torch::TensorOptions().dtype(torch::kFloat32)
                                  .requires_grad(false));
            int nAdvance = nSamplesInWindow;
            int nWindows = 0;
            for (int batch=0; batch<batchSize; ++batch)
            {
                int i1 = iStart + batch*nAdvance
                       + threadID*batchSize*nAdvance;
                int i2 = i1 + nSamplesInWindow;
                if (i2 > nSamples){break;}
                int j1 = 3*batch*nSamplesInWindow;
                float *xPtr = X.data_ptr<float> () + j1;
                rescaleAndCopy(nSamplesInWindow,
                               vertical+i1, north+i1, east+i1, xPtr);
                nWindows = nWindows + 1;
            }
            nCopy = nWindows*nSamplesInWindow;
            assert(nCopy + iStart + threadID*batchSize*nAdvance <= nSamples);
            auto XGPU = X.to({torch::kCUDA, deviceID});
            auto p = networks[threadID].forward(XGPU, applySigmoid);
            //std::cout << p.device().index() << std::endl;
            auto pHost = p.to(torch::kCPU);
            float *pPtr = pHost.data_ptr<float> ();
            copy(nCopy, pPtr, proba + iStart + threadID*batchSize*nAdvance);
            }
            iStart = iStart + nCopy;
            lastIndex = iStart;// + nCopy;
            if (iStart + nSamplesInWindow > nSamples){break;}
        }
    }
    else
    {
        predictProbability(nSamplesInWindow, vertical, north, east, &proba);
        int nHalf = nSamplesInWindow/2;
        int iDst = nHalf + nCenter;
        int iSrc = 2*nCenter;
        int nAdvanceWindow = 2*nCenter;
        //auto X = torch::zeros({batchSize, 3, nSamplesInWindow},
        //                       torch::TensorOptions().dtype(torch::kFloat32)
        //                      .requires_grad(false));
        for (int kloop=0; kloop<nSamples; ++kloop)
        {
            int nWindows = 0;
/*
            #pragma omp parallel \
             shared(batchSize, iDst, iSrc, nHalf, nAdvanceWindow, nDevices) \
             shared(std::cout, vertical, north, east, proba, deviceIDs) \
             shared(nCenter, networks, nSamples, nSamplesInWindow) \
             shared(torch::kCPU, torch::kFloat32, torch::kCUDA) \
             default(none) \
             reduction(+:nWindows)
*/
            {
            int threadID = 0;
#if defined(_OPENMP)
            threadID = omp_get_thread_num();
#endif
            int deviceID = deviceIDs[threadID];
            auto X = torch::zeros({batchSize, 3, nSamplesInWindow},
                                   torch::TensorOptions().dtype(torch::kFloat32)
                                  .requires_grad(false));

            for (int batch=0; batch<batchSize; ++batch)
            {
                int i1 = iSrc + batch*nAdvanceWindow
                       + threadID*batchSize*nAdvanceWindow;
                int i2 = i1 + nSamplesInWindow;
                if (i2 > nSamples){break;}
                int j1 = 3*batch*nSamplesInWindow;
                float *xPtr = X.data_ptr<float> () + j1;
                rescaleAndCopy(nSamplesInWindow,
                               vertical+i1, north+i1, east+i1, xPtr);
                nWindows = nWindows + 1;
            }
            auto XGPU = X.to({torch::kCUDA, deviceID}); //X.to({torch::kCUDA, 0});
            auto p = networks[threadID].forward(XGPU, applySigmoid);
            auto pHost = p.to(torch::kCPU);
            float *pPtr = pHost.data_ptr<float> ();
            for (int iwin=0; iwin<nWindows; iwin++)
            {
                int jSrc = nHalf - nCenter + iwin*nSamplesInWindow;
                int jDst = iDst + iwin*nAdvanceWindow
                         + threadID*batchSize*nAdvanceWindow;
                copy(nAdvanceWindow, pPtr + jSrc, proba + jDst);
            }

            } // End parallel

            iSrc = iSrc + nAdvanceWindow*nWindows;
            iDst = iDst + nAdvanceWindow*nWindows;
            lastIndex = iDst;
            if (iSrc + nSamplesInWindow > nSamples){break;}
        }
    }
    int nRemainder = nSamples - lastIndex;
    if (nRemainder > 0)
    {
        int i1 = nSamples - nSamplesInWindow;
        auto Xlast = torch::zeros({1, 3, nSamplesInWindow},
                                   torch::TensorOptions().dtype(torch::kFloat32)
                                  .requires_grad(false));
        float *xPtr = Xlast.data_ptr<float> ();
        rescaleAndCopy(nSamplesInWindow,
                       vertical+i1, north+i1, east+i1, xPtr);
        int j1 = nSamplesInWindow - nRemainder;
        auto XGPU = Xlast.to({torch::kCUDA, 0});
        auto p = pImpl->mNetworks[0].forward(XGPU, applySigmoid);
        auto pHost = p.to(torch::kCPU);
        float *pPtr = pHost.data_ptr<float> ();
        copy(nRemainder, pPtr + j1, proba + lastIndex);
    }
#if defined(_OPENMP)
    omp_set_num_threads(nOriginalThreads);
#endif
}
/// Determines if the model coefficients were loaded
template<UUSS::Device E>
bool Model<E>::haveModelCoefficients() const noexcept
{
    return pImpl->mHaveCoefficients;
}

/// Gets the minimum seismogram length
template<UUSS::Device E>
int Model<E>::getMinimumSeismogramLength() const noexcept
{
    return pImpl->mMinimumSeismogramLength;
}

/// Determines if the seismogram length is okay
template<UUSS::Device E>
bool Model<E>::isValidSeismogramLength(
    const int nSamples) const noexcept
{
    if (nSamples < getMinimumSeismogramLength()){return false;}
    if (nSamples%16 != 0){return false;}
    return true;
}

/// Determines the number of devices
template<UUSS::Device E>
int Model<E>::getNumberOfDevices() const noexcept
{
    return static_cast<int> (pImpl->mDeviceIDs.size());
}

/// Return the number of channels
template<UUSS::Device E>
int Model<E>::getInputNumberOfChannels() const noexcept
{
    return pImpl->mNetworks[0].mInputChannels;
}

//-----------------------------------------------------------------------------//
//                             Template instantiation                          //
//-----------------------------------------------------------------------------//
template class UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::GPU>;
template class UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::CPU>;
