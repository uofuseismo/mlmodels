#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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
    //HDF5Loader loader; 
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
    // Compare with python - I think dmax is like 0.009 which is plenty 
    // sufficient for this activity
    auto dmax = infinityNorm(nSamples+nExtra, proba.data(), probaRef.data());
    std::cout << dmax << std::endl;
    EXPECT_NEAR(dmax, 0, 1.e-2);
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
}

TEST(ThreeComponentPicker, ZRUNetGPU)
{
    try
    {
        UUSS::ThreeComponentPicker::ZRUNet::Model<UUSS::Device::GPU> model;
        EXPECT_NO_THROW(
            model.loadWeightsFromHDF5("../testing/models/test_zrunet_p.h5"));
    }
    catch (const std::exception &e)
    {
        return;
    }
}

}
