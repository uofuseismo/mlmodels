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
    EXPECT_NO_THROW(polarity.setPolarityThreshold(tol));
    EXPECT_NEAR(polarity.getPolarityThreshold(), tol, 1.e-8);
/*
try
{
polarity.loadWeightsFromPT("/data/machineLearning/rossFirstMotion/finetuned_models/models_005_scripted.pt");
}
catch (const std::exception &e)
{
 std::cerr << e.what() << std::endl;
}
*/
    EXPECT_NO_THROW(
        polarity.loadWeightsFromHDF5("../testing/models/test_fmnet.h5"));
    std::vector<float> ex1, ex2, ex3, ex4, ex5;
    loadTextFile("../testing/data/fmnet_test_inputs.txt",
                 ex1, ex2, ex3, ex4, ex5);
    std::vector<float> pUp(5, 0), pDown(5, 0), pUnknown(5, 0);
    std::vector<float> pRefUp({0.0008, 0.0179, 0.0739, 0.5519, 0.0029});
    std::vector<float> pRefDown({0.6310, 0.0350, 0.0029, 0.0008, 0.7787});
    std::vector<float> pRefUnknown({0.3682, 0.9471, 0.9233, 0.4473, 0.2185});
    EXPECT_NO_THROW(
        polarity.predictProbability(ex1.size(), ex1.data(),
                                    &pUp[0], &pDown[0], &pUnknown[0]));
    EXPECT_NO_THROW(
        polarity.predictProbability(ex2.size(), ex2.data(),
                                    &pUp[1], &pDown[1], &pUnknown[1]));
    EXPECT_NO_THROW(
        polarity.predictProbability(ex3.size(), ex3.data(),
                                    &pUp[2], &pDown[2], &pUnknown[2]));
    EXPECT_NO_THROW(
        polarity.predictProbability(ex4.size(), ex4.data(), &pUp[3],
                                    &pDown[3], &pUnknown[3]));
    EXPECT_NO_THROW(
        polarity.predictProbability(ex5.size(), ex5.data(),
                                    &pUp[4], &pDown[4], &pUnknown[4]));
    for (int i=0; i<static_cast<int> (pRefUp.size()); ++i)
    {
        EXPECT_NEAR(pRefUp[i], pUp[i], 1.e-3);
        EXPECT_NEAR(pRefDown[i], pDown[i], 1.e-3);
        EXPECT_NEAR(pRefUnknown[i], pUnknown[i], 1.e-3);
    }
int result = polarity.predict(ex5.size(), ex5.data());
int down =-1;
#ifdef CHPC
    EXPECT_NEAR(result, down, 1.e-10);
#else
    EXPECT_EQ(result, down);
#endif
//std::cout << pUp << " " << pDown << " " << pUnknown << std::endl;

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
