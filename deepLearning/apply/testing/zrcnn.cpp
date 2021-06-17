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
std::cout << picker.predict(ex1.size(), ex1.data()) << std::endl;
std::cout << picker.predict(ex2.size(), ex2.data()) << std::endl;
std::cout << picker.predict(ex3.size(), ex3.data()) << std::endl;
std::cout << picker.predict(ex4.size(), ex4.data()) << std::endl;
}

}
