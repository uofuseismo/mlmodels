#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "uuss/oneComponentPicker/zcnn/model.hpp"
#include "uuss/oneComponentPicker/zcnn/processData.hpp"
#include "uuss/threeComponentPicker/zcnn/processData.hpp"
#include "uuss/threeComponentPicker/zcnn/model.hpp"
#include <gtest/gtest.h>

namespace
{

void loadTextFile(const std::string &textFile,
                  std::vector<double> &ex1,
                  std::vector<double> &ex2,
                  std::vector<double> &ex3,
                  std::vector<double> &ex4)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    ex1.resize(400, 0);
    ex2.resize(400, 0); 
    ex3.resize(400, 0);
    ex4.resize(400, 0);
    int i = 0;
    while (std::getline(infile, line))
    {   
        double v1, v2, v3, v4;
        sscanf(line.c_str(), "%lf,%lf,%lf,%lf\n", &v1, &v2, &v3, &v4);
        ex1.at(i) = v1;
        ex2.at(i) = v2;
        ex3.at(i) = v3;
        ex4.at(i) = v4;
        i = i + 1;
    }
}

void loadTextFile(const std::string &textFile,
                  std::vector<double> &z1,
                  std::vector<double> &n1,
                  std::vector<double> &e1,
                  std::vector<double> &z2,
                  std::vector<double> &n2,
                  std::vector<double> &e2,
                  std::vector<double> &z3,
                  std::vector<double> &n3,
                  std::vector<double> &e3,
                  std::vector<double> &z4,
                  std::vector<double> &n4,
                  std::vector<double> &e4)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    z1.resize(600, 0);
    n1.resize(600, 0);
    e1.resize(600, 0);
    z2.resize(600, 0);
    n2.resize(600, 0);
    e2.resize(600, 0);
    z3.resize(600, 0);
    n3.resize(600, 0);
    e3.resize(600, 0);
    z4.resize(600, 0);
    n4.resize(600, 0);
    e4.resize(600, 0);
    int i = 0;
    while (std::getline(infile, line))
    {   
        double vz1, vn1, ve1;
        double vz2, vn2, ve2;
        double vz3, vn3, ve3;
        double vz4, vn4, ve4;
        sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
               &vz1, &vn1, &ve1,
               &vz2, &vn2, &ve2,
               &vz3, &vn3, &ve3,
               &vz4, &vn4, &ve4);
        z1.at(i) = vz1; 
        n1.at(i) = vn1; 
        e1.at(i) = ve1;
        z2.at(i) = vz2;  
        n2.at(i) = vn2;  
        e2.at(i) = ve2;
        z3.at(i) = vz3;  
        n3.at(i) = vn3;  
        e3.at(i) = ve3;
        z4.at(i) = vz4;  
        n4.at(i) = vn4;  
        e4.at(i) = ve4;
        i = i + 1;
    }   
}


TEST(ZCNN, Preprocess)
{
    UUSS::OneComponentPicker::ZCNN::ProcessData process;
    EXPECT_NEAR(process.getTargetSamplingPeriod(), 0.01, 1.e-14);
}

TEST(ZCNN3C, Preprocess)
{
    UUSS::ThreeComponentPicker::ZCNN::ProcessData process;
    EXPECT_NEAR(process.getTargetSamplingPeriod(), 0.01, 1.e-14);
}

//../testing/models/test_zrcnnpick_p.h5

TEST(ZCNN, Picker)
{
    std::vector<double> ex1, ex2, ex3, ex4;
    loadTextFile("../testing/data/p_signals_zcnn_p.csv", ex1, ex2, ex3, ex4);
    UUSS::OneComponentPicker::ZCNN::ProcessData process;
    UUSS::OneComponentPicker::ZCNN::Model picker;
    EXPECT_EQ(picker.getSignalLength(), 400);
    EXPECT_NEAR(picker.getSamplingPeriod(), 0.01, 1.e-14);
    EXPECT_NEAR(picker.getSamplingPeriod(),
                process.getTargetSamplingPeriod(), 1.e-14);
    EXPECT_NO_THROW(picker.loadWeightsFromHDF5(
        "../testing/models/test_zrcnnpick_p.h5"));
    auto p1 = picker.predict(ex1.size(), ex1.data());
    auto p2 = picker.predict(ex2.size(), ex2.data());
    auto p3 = picker.predict(ex3.size(), ex3.data());
    auto p4 = picker.predict(ex4.size(), ex4.data());
    EXPECT_NEAR(0.3296588,  p1, 1.e-5);
    EXPECT_NEAR(0.29644084, p2, 1.e-5);
    EXPECT_NEAR(0.43443173, p3, 1.e-5);
    EXPECT_NEAR(0.44393498, p4, 1.e-5);
    std::vector<double> reference({0.3296588, 0.29644084,
                                   0.43443173, 0.44393498});
    int nSignals = 36;
    int batchSize = 32;
    std::vector<double> signals(nSignals*ex1.size(), 0);
    for (int i = 0; i < nSignals; ++i)
    {
        auto i1 = i*ex1.size();
        if (i%4 == 0)
        {
            std::copy(ex1.data(), ex1.data() + ex1.size(),
                      signals.data() + i1);  
        }
        else if (i%4 == 1)
        {
            std::copy(ex2.data(), ex2.data() + ex2.size(),
                      signals.data() + i1);
        }
        else if (i%4 == 2)
        {
            std::copy(ex3.data(), ex3.data() + ex3.size(),
                      signals.data() + i1);
        }
        else if (i%4 == 3)
        {
            std::copy(ex4.data(), ex4.data() + ex4.size(),
                      signals.data() + i1);
        }
        else
        {
            std::cerr << "Shouldn't be here" << std::endl;
        } 
    }
    std::vector<double> perts(nSignals);
    auto pPtr = perts.data();
    picker.predict(nSignals, ex1.size(), signals.data(), &pPtr, batchSize); 
    for (int i = 0; i < nSignals; ++i)
    {
        EXPECT_NEAR(reference[i%4], perts[i], 1.e-5);
    }
}

TEST(ZCNN3C, Picker)
{
    std::vector<double> z1, n1, e1, z2, n2, e2, z3, n3, e3, z4, n4, e4;
    loadTextFile("../testing/data/s_signals_zcnn3c_s.csv",
                 z1, n1, e1,
                 z2, n2, e2,
                 z3, n3, e3,
                 z4, n4, e4);
    UUSS::ThreeComponentPicker::ZCNN::ProcessData process;
    UUSS::ThreeComponentPicker::ZCNN::Model picker;
    EXPECT_EQ(picker.getSignalLength(), 600);
    EXPECT_EQ(z1.size(), 600);
    EXPECT_NEAR(picker.getSamplingPeriod(), 0.01, 1.e-14);
    EXPECT_NEAR(picker.getSamplingPeriod(),
                process.getTargetSamplingPeriod(), 1.e-14);
    EXPECT_NO_THROW(picker.loadWeightsFromHDF5(
        "../testing/models/test_zrcnnpick3c_s.h5"));
    std::array<double, 4> yRef{0.5312093, -0.42324576,  0.55321896, 0.1556777};
    auto y1 = picker.predict(z1.size(), z1.data(), n1.data(), e1.data());
    auto y2 = picker.predict(z2.size(), z2.data(), n2.data(), e2.data());
    auto y3 = picker.predict(z3.size(), z3.data(), n3.data(), e3.data());
    auto y4 = picker.predict(z4.size(), z4.data(), n4.data(), e4.data());
    EXPECT_NEAR(y1, yRef[0], 1.e-4);
    EXPECT_NEAR(y2, yRef[1], 1.e-4);
    EXPECT_NEAR(y3, yRef[2], 1.e-4);
    EXPECT_NEAR(y4, yRef[3], 1.e-4);
   
    int ns = 36;
    int batchSize = 32;
    std::vector<double> zSignals(ns*z1.size(), 0); 
    std::vector<double> nSignals(ns*z1.size(), 0); 
    std::vector<double> eSignals(ns*z1.size(), 0); 
    for (int i = 0; i < ns; ++i)
    {   
        auto i1 = i*z1.size();
        if (i%4 == 0)
        {
            std::copy(z1.data(), z1.data() + z1.size(),
                      zSignals.data() + i1);
            std::copy(n1.data(), n1.data() + n1.size(),
                      nSignals.data() + i1);
            std::copy(e1.data(), e1.data() + e1.size(),
                      eSignals.data() + i1);
        }
        else if (i%4 == 1)
        {
            std::copy(z2.data(), z2.data() + z2.size(),
                      zSignals.data() + i1);
            std::copy(n2.data(), n2.data() + n2.size(),
                      nSignals.data() + i1);
            std::copy(e2.data(), e2.data() + e2.size(),
                      eSignals.data() + i1);
        }
        else if (i%4 == 2)
        {
            std::copy(z3.data(), z3.data() + z3.size(),
                      zSignals.data() + i1); 
            std::copy(n3.data(), n3.data() + n3.size(),
                      nSignals.data() + i1); 
            std::copy(e3.data(), e3.data() + e3.size(),
                      eSignals.data() + i1);
        }
        else if (i%4 == 3)
        {
            std::copy(z4.data(), z4.data() + z4.size(),
                      zSignals.data() + i1); 
            std::copy(n4.data(), n4.data() + n4.size(),
                      nSignals.data() + i1); 
            std::copy(e4.data(), e4.data() + e4.size(),
                      eSignals.data() + i1);
        }
        else
        {
            std::cerr << "Shouldn't be here" << std::endl;
        }
    }
    std::vector<double> perts(ns);
    auto pPtr = perts.data();
    picker.predict(ns, z1.size(),
                   zSignals.data(),
                   nSignals.data(),
                   eSignals.data(),
                   &pPtr, batchSize); 
    for (int i = 0; i < ns; ++i)
    {   
        EXPECT_NEAR(yRef[i%4], perts[i], 1.e-5);
    }
}

}
