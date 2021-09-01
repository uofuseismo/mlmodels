#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "uuss/forwardSimulation/eikonet/model.hpp"
#include <gtest/gtest.h>

namespace
{

using namespace UUSS::ForwardSimulation::EikoNet;

TEST(EikonNet, cpu)
{
    Model model;
    std::tuple<double, double, double> source{0, 0, 0};
    std::tuple<double, double, double> receiver{1, 1, 1};
 std::cout << "start" << std::endl;
    auto t = model.computeTravelTime(source, receiver);
std::cout << "end" << " " << t << std::endl;
for (int i = 0; i < 30; ++i)
{
    t = model.computeTravelTime(source, receiver);
}
std::cout << "end again" << " " << t << std::endl;
}


}

