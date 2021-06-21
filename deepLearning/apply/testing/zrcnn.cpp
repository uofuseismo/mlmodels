#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "uuss/oneComponentPicker/zcnn/model.hpp"
#include "uuss/oneComponentPicker/zcnn/processData.hpp"
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

TEST(ZCNN, Preprocess)
{
    UUSS::OneComponentPicker::ZCNN::ProcessData process;
}

//../testing/models/test_zrcnnpick_p.h5

TEST(ZCNN, Picker)
{
    std::vector<double> ex1, ex2, ex3, ex4;
    loadTextFile("../testing/data/p_signals_zcnn_p.csv", ex1, ex2, ex3, ex4);
    UUSS::OneComponentPicker::ZCNN::ProcessData process;
    UUSS::OneComponentPicker::ZCNN::Model picker;
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
            EXPECT_TRUE(false);
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

}
