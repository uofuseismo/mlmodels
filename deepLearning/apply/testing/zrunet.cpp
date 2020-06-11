#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "uuss/threeComponentPicker/zrunet/processData.hpp"
#include "uuss/threeComponentPicker/zrunet/model.hpp"
#include <gtest/gtest.h>

namespace
{

std::vector<float> loadTextFile(const std::string &textFile)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    std::vector<float> res;
    res.reserve(360000);
    while (std::getline(infile, line))
    {
        double t, v;
        sscanf(line.c_str(), "%lf, %lf\n", &t, &v);
        res.push_back(static_cast<float> (v));
    }
    return res;
}

template<class T>
T infinityNorm(const int n, const T x[], const T y[])
{
    T dmax = 0;
    #pragma omp simd reduction(max:dmax)
    for (int i=0; i<n; ++i)
    {
        dmax = std::max(std::abs(x[i] - y[i]), dmax);
    }
    return dmax;
}

TEST(ThreeComponentPicker, ProcessData)
{
    // Load data
    auto vTrace = loadTextFile("../testing/data/PB.B206.EHZ.zrunet_p.txt");
    auto nTrace = loadTextFile("../testing/data/PB.B206.EH1.zrunet_p.txt");
    auto eTrace = loadTextFile("../testing/data/PB.B206.EH2.zrunet_p.txt");
    // Load reference
    auto vRef = loadTextFile("../testing/data/PB.B206.EHZ.PROC.zrunet_p.txt");
    auto nRef = loadTextFile("../testing/data/PB.B206.EH1.PROC.zrunet_p.txt");
    auto eRef = loadTextFile("../testing/data/PB.B206.EH2.PROC.zrunet_p.txt");
    // Traces must be processed prior to model application
    const double dt = 1.0/100.0;
    UUSS::ThreeComponentPicker::ZRUNet::ProcessData process;
    // Traes must be processed prior to model application
    std::vector<float> temp(vTrace);
    process.processWaveform(temp.size(), dt, temp.data(), &vTrace);
    temp = nTrace;
    process.processWaveform(temp.size(), dt, temp.data(), &nTrace);
    temp = eTrace;
    process.processWaveform(temp.size(), dt, temp.data(), &eTrace);
    // Compare
    auto dmax1 = infinityNorm(vTrace.size(), vRef.data(), vTrace.data());
    auto dmax2 = infinityNorm(nTrace.size(), nRef.data(), nTrace.data());
    auto dmax3 = infinityNorm(eTrace.size(), eRef.data(), eTrace.data());
    // Some numbers are in the 10,000's so this is fine for floating 
    // precision
    EXPECT_NEAR(dmax1, 0, 1.e-1);
    EXPECT_NEAR(dmax2, 0, 1.e-1);
    EXPECT_NEAR(dmax3, 0, 1.e-1);
}

TEST(ThreeComponentPicker, ZRUnetCPU)
{
    UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::CPU> picker;
    EXPECT_NO_THROW(picker.loadWeightsFromHDF5("../testing/models/test_zrunet_p.h5"));
    // Load data
    auto vTrace = loadTextFile("../testing/data/PB.B206.EHZ.zrunet_p.txt");
    auto nTrace = loadTextFile("../testing/data/PB.B206.EH1.zrunet_p.txt");
    auto eTrace = loadTextFile("../testing/data/PB.B206.EH2.zrunet_p.txt");
    auto probaRef = loadTextFile("../testing/data/PB.UNET.P.PYTHON.zrunet.txt");
    // Traces must be processed prior to model application 
    const double dt = 1.0/100.0;
    UUSS::ThreeComponentPicker::ZRUNet::ProcessData process;
    // Traes must be processed prior to model application
    std::vector<float> temp(vTrace);
    process.processWaveform(temp.size(), dt, temp.data(), &vTrace);
    temp = nTrace;
    process.processWaveform(temp.size(), dt, temp.data(), &nTrace);
    temp = eTrace;
    process.processWaveform(temp.size(), dt, temp.data(), &eTrace);
    // Do a small amount of data so this test doesn't take forever
    std::vector<float> proba(vTrace.size(), 0);
    int nSamples = 1008*50;
    int nExtra = 50;
    int segment = 1008;
    for (int i=0; i<nSamples; i=i+segment)
    { 
        const float *vPtr = vTrace.data() + i;
        const float *nPtr = nTrace.data() + i;
        const float *ePtr = eTrace.data() + i;
        float *pPtr = proba.data() + i;
        if (i+segment > static_cast<int> (vTrace.size())){break;}
        EXPECT_NO_THROW(picker.predictProbability(segment,
                                                  vPtr, nPtr, ePtr, &pPtr));
    }
    // Do one more segment to check edge case
    const float *vPtr = vTrace.data() + nSamples - (segment - nExtra);
    const float *nPtr = nTrace.data() + nSamples - (segment - nExtra);
    const float *ePtr = eTrace.data() + nSamples - (segment - nExtra);
    std::vector<float> probaExtra(nSamples, 100); // Check for uninitialized
    float *pPtr = probaExtra.data(); //proba.data() + nSamples;
    EXPECT_NO_THROW(picker.predictProbability(segment,
                                              vPtr, nPtr, ePtr, &pPtr));
    std::copy(probaExtra.data() + segment - nExtra, probaExtra.data() + segment,
              proba.data() + nSamples);
    // Compare with python - I think dmax is like 0.0003 which is plenty .
    // The first pass dmax was 0.009.  This is because PyTorch's default API
    // switched the epsilon in the batch normalization.
    auto dmax = infinityNorm(nSamples+nExtra, proba.data(), probaRef.data());
    //for (int i=7500; i<7600; ++i)
    //{
    //      std::cout << proba[i] << "," << probaRef[i]
    //                << "," << proba[i] - probaRef[i] << std::endl;
    //}
    EXPECT_NEAR(dmax, 0, 5.e-4);
    // Run batched variant with no overlap
    int batchSize = 8; // Good on a cpu
    std::vector<float> probaNew(nSamples + nExtra, 100);
    pPtr = probaNew.data();
    EXPECT_NO_THROW(
        picker.predictProbability(nSamples+nExtra, segment, 0,
                                  vTrace.data(), nTrace.data(), eTrace.data(),
                                  &pPtr, batchSize));
    // Ensure self-consistent
    dmax = infinityNorm(nSamples+nExtra, proba.data(), probaNew.data());
    EXPECT_NEAR(dmax, 0, 1.e-6);

    // Repeat for overlapping windows
    int nCenter = 250;
    std::fill(proba.begin(), proba.end(), 0);
    pPtr = proba.data();
    picker.predictProbability(segment,
                              vTrace.data(), nTrace.data(), eTrace.data(),
                              &pPtr);
    // We have now computed:
    // [1008/2 - 250:1008/2 + 250] = [254:754].
    // Additionally, we treat the first 254 samples as an edge case and
    // simply copy.  Hence, we want to insert the results from the next
    // iteration at center = 504 + nCenter = 754.  Conseqeuntly, this
    // will copy [754:1254] to the probability vector - i.e., the next
    // 500 samples..
    int iDst = segment/2 + nCenter;
    // For this to work we need to work need our window centered at
    // segment/2 + 2*nCenter = 1004 (half way between 754 and 1254.
    // The segment around this point, 1004, is +/- segment/2.  Hence,
    // we start at segment/2 + 2*ncenter - segment/2 = 2*nCenter
    int iSrc = 2*nCenter;
    std::vector<float> pWork(segment, 0);
    while (true)
    {
        if (iSrc + segment > nSamples){break;}
        const float *vPtr = vTrace.data() + iSrc;
        const float *nPtr = nTrace.data() + iSrc;
        const float *ePtr = eTrace.data() + iSrc;
        float *pWorkPtr = pWork.data();
        picker.predictProbability(segment, vPtr, nPtr, ePtr, &pWorkPtr);
        int j1 = segment/2 - nCenter;
        for (int j=0; j<2*nCenter; ++j)
        {
            proba.at(iDst + j) = pWork.at(j1 + j);
        }
        //printf("%d, %d, %d\n", j1, iSrc, iDst);
        iSrc = iSrc + 2*nCenter;
        iDst = iDst + 2*nCenter;
    }
    // Add one more segment so I can check an edge case
    vPtr = vTrace.data() + nSamples - segment;
    nPtr = nTrace.data() + nSamples - segment;
    ePtr = eTrace.data() + nSamples - segment;
    std::fill(probaExtra.begin(), probaExtra.end(), 0);
    pPtr = probaExtra.data(); //proba.data() + nSamples;
    picker.predictProbability(segment, vPtr, nPtr, ePtr, &pPtr);
    int lastIndex = iDst;
    int nCopy = nSamples - lastIndex;
    std::copy(probaExtra.data() + segment - nCopy, probaExtra.data() + segment,
              proba.data() + lastIndex);
    // Repeat with library algorithm
    std::fill(probaNew.begin(), probaNew.end(), 100);
    pPtr = probaNew.data();
    EXPECT_NO_THROW(
        picker.predictProbability(nSamples, segment, nCenter,
                                  vTrace.data(), nTrace.data(), eTrace.data(),
                                  &pPtr, batchSize));
    // And compare
    dmax = infinityNorm(nSamples, proba.data(), probaNew.data());
    EXPECT_NEAR(dmax, 0, 1.e-6);
}

TEST(ThreeComponentPicker, ZRUNetGPU)
{
    // Skip ahead and get pre-processed data in here
    auto vTrace = loadTextFile("../testing/data/PB.B206.EHZ.PROC.zrunet_p.txt");
    auto nTrace = loadTextFile("../testing/data/PB.B206.EH1.PROC.zrunet_p.txt");
    auto eTrace = loadTextFile("../testing/data/PB.B206.EH2.PROC.zrunet_p.txt");
    // By this point we've established that the library algorithms work.  
    // Consequently, we test the GPU implementation against the reference 
    // CPU implementation.
    UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::CPU> cpuPicker;
    EXPECT_NO_THROW(
        cpuPicker.loadWeightsFromHDF5("../testing/models/test_zrunet_p.h5"));
    // Create two test scenarios
    int nSamples = 1008*50;
    int nExtra = 50;
    int segment = 1008; 
    int cpuBatchSize = 4;
    std::vector<float> probaRef1(nSamples + nExtra, 0);
    float *pPtr = probaRef1.data();
    EXPECT_NO_THROW(
        cpuPicker.predictProbability(nSamples + nExtra, segment, 0,
                                    vTrace.data(), nTrace.data(), eTrace.data(),
                                    &pPtr, cpuBatchSize));

    int nCenter = 250;
    std::vector<float> probaRef2(nSamples, 0);
    pPtr = probaRef2.data();
    EXPECT_NO_THROW(
        cpuPicker.predictProbability(nSamples, segment, nCenter,
                                    vTrace.data(), nTrace.data(), eTrace.data(),
                                    &pPtr, cpuBatchSize));
    float error1 = 0;
    float error2 = 0;
    try
    {
        UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::GPU> gpuPicker;
        EXPECT_NO_THROW(
            gpuPicker.loadWeightsFromHDF5("../testing/models/test_zrunet_p.h5"));

        int gpuBatchSize = 64;
        std::vector<float> proba1(nSamples + nExtra, 0);
        pPtr = proba1.data();
        EXPECT_NO_THROW(
            gpuPicker.predictProbability(nSamples + nExtra, segment, 0,
                                         vTrace.data(), nTrace.data(),
                                         eTrace.data(), &pPtr, gpuBatchSize));
        error1 = infinityNorm(proba1.size(), probaRef1.data(), proba1.data());
        EXPECT_NEAR(error1, 0, 1.e-5);
 
        std::vector<float> proba2(nSamples, 0);
        pPtr = proba2.data();
        EXPECT_NO_THROW(
            gpuPicker.predictProbability(nSamples, segment, nCenter,
                                         vTrace.data(), nTrace.data(),
                                         eTrace.data(), &pPtr, gpuBatchSize));
        error2 = infinityNorm(proba2.size(), probaRef2.data(), proba2.data());
        EXPECT_NEAR(error2, 0, 1.e-5);

    }
    catch (const std::exception &e)
    {
        return;
    }
}

}
