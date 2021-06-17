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
                  std::vector<float> &ex1,
                  std::vector<float> &ex2)
{
    std::ifstream infile(textFile, std::ios::in);
    std::string line;
    ex1.resize(400, 0);
    ex2.resize(400, 0); 
    int i = 0;
    while (std::getline(infile, line))
    {   
        double v1, v2;
        sscanf(line.c_str(), "%lf, %lf\n", &v1, &v2);
        ex1.at(i) = v1;
        ex2.at(i) = v2;
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
    UUSS::OneComponentPicker::ZCNN::ProcessData process;
    UUSS::OneComponentPicker::ZCNN::Model picker;
    EXPECT_NO_THROW(picker.loadWeightsFromHDF5(
        "../testing/models/test_zrcnnpick_p.h5", true));
}

}
