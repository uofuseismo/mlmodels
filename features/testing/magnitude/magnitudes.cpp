#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "uuss/features/magnitude/verticalChannelFeatures.hpp"
#include <gtest/gtest.h>

namespace
{

void readSeismograms(const std::string &fileName,
                     std::vector<double> *a, std::vector<double> *v)
{
    std::ifstream infl(fileName, std::ios::in);
    a->clear();
    v->clear();
    if (infl.is_open())
    {
        std::string line;
        a->reserve(9112);
        v->reserve(9112);
        while (std::getline(infl, line))
        {
            double t, ai, vi;
            sscanf(line.c_str(), "%lf, %lf, %lf\n", &t, &ai, &vi);
            a->push_back(ai);
            v->push_back(vi);
        }
        infl.close();
    }
}


using namespace UUSS::Features::Magnitude;

TEST(FeaturesMagnitude, VerticalChannelFeatures)
{
    std::vector<double> acceleration, velocity;
    double startTime{30.52};
    readSeismograms("data/sru_enz_hhz.txt", &acceleration, &velocity);

    VerticalChannelFeatures features;
    double samplingRate{100}; 
    const std::string velocityUnits{"DU/M/S"};
    const std::string accelerationUnits{"DU/M/S**2"};
    const double velocitySimpleResponse{5.617686e+08};
    const double accelerationSimpleResponse{1.854997e+05};
 
    EXPECT_NO_THROW(features.initialize(samplingRate,
                                        velocitySimpleResponse,
                                        velocityUnits));
    EXPECT_TRUE(features.isInitialized());
    EXPECT_NEAR(features.getSamplingRate(), samplingRate, 1.e-14);
    EXPECT_NEAR(features.getSimpleResponse(), velocitySimpleResponse, 1.e-4);
    EXPECT_EQ(features.getSimpleResponseUnits(), velocityUnits);

    EXPECT_NEAR(features.getTargetSamplingRate(), 100, 1.e-14);
    EXPECT_NEAR(features.getTargetSamplingPeriod(), 1./100, 1.e-14);

    EXPECT_NO_THROW(features.process(velocity, startTime));
    auto velocitySignal = features.getVelocitySignal();
    //------------------------------------------------------------------------// 
    EXPECT_NO_THROW(features.initialize(samplingRate,
                                        accelerationSimpleResponse,
                                        accelerationUnits));
    EXPECT_NEAR(features.getSimpleResponse(), accelerationSimpleResponse,1.e-4);
    EXPECT_EQ(features.getSimpleResponseUnits(), accelerationUnits);

    EXPECT_NO_THROW(features.process(acceleration, startTime));
    auto velocitySignal2 = features.getVelocitySignal();

    auto ofl = std::ofstream("vel.txt");
    for (int i = 0; i < velocitySignal.size(); ++i)
    {
        ofl << i*0.01 << " " << velocitySignal[i] << " " << velocitySignal2[i] << std::endl;
    }
    ofl.close();
}

}
