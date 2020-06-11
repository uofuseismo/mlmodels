#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "uuss/firstMotion/fmnet/processData.hpp"
#include "uuss/firstMotion/fmnet/model.hpp"
#include <gtest/gtest.h>

namespace
{

void loadTextFile(const std::string &textFile,
                  std::vector<float> &ex1,
                  std::vector<float> &ex2,
                  std::vector<float> &ex3,
                  std::vector<float> &ex4,
                  std::vector<float> &ex5)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    ex1.resize(400, 0);
    ex2.resize(400, 0);
    ex3.resize(400, 0);
    ex4.resize(400, 0);
    ex5.resize(400, 0);
    int i = 0;
    while (std::getline(infile, line))
    {
        double v1, v2, v3, v4, v5;
        sscanf(line.c_str(), "%lf, %lf, %lf, %lf, %lf\n",
               &v1, &v2, &v3, &v4, &v5);
        ex1[i] = v1;
        ex2[i] = v2;
        ex3[i] = v3;
        ex4[i] = v4;
        ex5[i] = v5;
        i = i + 1;
    }
}

TEST(FMNet, Preprocess)
{
    //UUSS::FirstMotion::FMNet::ProcessData process; 
}

TEST(FMNet, FMNetCPU)
{
    UUSS::FirstMotion::FMNet::Model<UUSS::Device::CPU> polarity;
    double tol = 0.4;
    EXPECT_EQ(polarity.getSignalLength(), 400); 
    EXPECT_NO_THROW(polarity.setPolarityThreshold(tol));
    EXPECT_NEAR(polarity.getPolarityThreshold(), tol, 1.e-8);
    EXPECT_NO_THROW(
        polarity.loadWeightsFromHDF5("../testing/models/test_fmnet.h5"));
    std::vector<float> ex1, ex2, ex3, ex4, ex5;
    loadTextFile("../testing/data/fmnet_test_inputs.txt",
                 ex1, ex2, ex3, ex4, ex5);
    float pUp, pDown, pUnknown;
    EXPECT_NO_THROW(
        polarity.predictProbability(ex1.size(),
                                    ex1.data(), &pUp, &pDown, &pUnknown));
std::cout << pUp << " " << pDown << " " << pUnknown << std::endl;
}

TEST(FMNet, FMNetGPU)
{
    try
    {
        UUSS::FirstMotion::FMNet::Model<UUSS::Device::GPU> polarity;
        EXPECT_NO_THROW(
            polarity.loadWeightsFromHDF5("../testing/models/test_fmnet.h5"));
    }
    catch (const std::exception &e)
    {
        std::cerr << "GPU likely not set - " << e.what() << std::endl;
    }
}

}
