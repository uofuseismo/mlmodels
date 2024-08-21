#include <string>
#include <vector>
#include <uussmlmodels/associators/phaseLink/inference.hpp>
#include <uussmlmodels/associators/phaseLink/pick.hpp>
#include <uussmlmodels/associators/phaseLink/arrival.hpp>
#include <gtest/gtest.h>

using namespace UUSSMLModels::Associators::PhaseLink;

namespace
{
std::string utahModel{"/home/bbaker/Codes/mlmodels/associators/phaseLink/models/utah.onnx"};

Pick toPick(int64_t identifier,
            const std::string &network,
            const std::string &station,
            const std::string &phase,
            const double time)
{
    Pick pick;
    pick.setIdentifier(identifier);
    pick.setNetwork(network);
    pick.setStation(station);
    pick.setTime(time);
    Pick::Phase iPhase{Pick::Phase::P};
    if (phase == "S"){iPhase = Pick::Phase::S;}
    pick.setPhase(iPhase);
    return pick;
}

std::vector<Pick> picksFor60000197() // Blast
{
    std::vector<Pick> picks{
        ::toPick(0, "UU", "CWU", "P", 1349131125.0860946),
        ::toPick(1, "UU", "MID", "P", 1349131125.1194382),
        ::toPick(2, "UU", "NOQ", "P", 1349131126.8506584),
        ::toPick(3, "UU", "WTU", "P", 1349131127.3981037),
        ::toPick(4, "UU", "GMU", "P", 1349131129.748004),
        ::toPick(5, "UU", "SAIU", "P", 1349131130.4178166),
        ::toPick(6, "UU", "RBU", "P", 1349131131.920422),
        ::toPick(7, "UU", "SNUT", "P", 1349131133.02),
        ::toPick(8, "UU", "DCU", "P", 1349131133.35),
        ::toPick(9, "UU", "NAIU", "P", 1349131133.38),
        ::toPick(10, "UU", "WMUT", "P", 1349131133.7190835),
        ::toPick(11, "UU", "NLU", "P", 1349131134.5819006),
        ::toPick(12, "UU", "JLU", "P", 1349131134.6785064),
        ::toPick(13, "US", "DUG", "P", 1349131135.1472971),
        ::toPick(14, "UU", "HLJ", "P", 1349131135.337707),
        ::toPick(15, "UU", "FPU", "P", 1349131135.5393994),
        ::toPick(16, "UU", "SUU", "P", 1349131136.976498),
        ::toPick(17, "UU", "RRCU", "P", 1349131137.23),
        ::toPick(18, "UU", "DAU", "P", 1349131137.767431),
        ::toPick(19, "UU", "SPU", "P", 1349131139.3414888),
        ::toPick(20, "UU", "GZU", "P", 1349131141.606655),
        ::toPick(21, "UU", "WVUT", "P", 1349131145.3714025),
        ::toPick(22, "UU", "FSU", "P", 1349131146.911738),
        ::toPick(23, "UU", "SNO", "P", 1349131149.0277183)
    };
    return picks;
}

std::vector<Pick> picksFor60000637()
{
    std::vector<Pick> picks{
        ::toPick(0, "UU", "KNB", "P", 1349579470.959896),
        ::toPick(1, "UU", "EKU", "P", 1349579471.1288135),
        ::toPick(2, "UU", "KNB", "S", 1349579473.809814),
        ::toPick(3, "UU", "ZNPU", "P", 1349579474.1981616),
        ::toPick(4, "UU", "PKCU", "P", 1349579475.0986772),
        ::toPick(5, "UU", "BHU", "P", 1349579475.5496635),
        ::toPick(6, "UU", "LCMT", "P", 1349579475.74),
        ::toPick(7, "UU", "SZCU", "P", 1349579476.5179846),
        ::toPick(8, "UU", "ZNPU", "S", 1349579479.486888),
        ::toPick(9, "UU", "BHU", "S", 1349579481.808445),
        ::toPick(10, "UU", "LCMT", "S", 1349579482.0595295),
        ::toPick(11, "UU", "ARUT", "P", 1349579482.8372333),
        ::toPick(12, "UU", "SZCU", "S", 1349579483.5194418),
        ::toPick(13, "UU", "MTPU", "P", 1349579484.9070563),
        ::toPick(14, "UU", "DWU", "P", 1349579485.1424851),
        ::toPick(15, "UU", "ICU", "P", 1349579485.2657948),
        ::toPick(16, "UU", "NMU", "P", 1349579491.7630632),
        ::toPick(17, "UU", "MSU", "P", 1349579492.3860478),
        ::toPick(18, "UU", "MMU", "P", 1349579494.4270148),
        ::toPick(19, "UU", "HMU", "P", 1349579498.615979)
    };
    return picks;
}

}

TEST(AssociatorsPhaseLink, Pick)
{
    Pick pick;
    pick.setNetwork("UU");
    pick.setStation("CWU");
    pick.setPhase(Pick::Phase::S);
    pick.setTime(1349131125.0860946);
    pick.setIdentifier(122);

    Pick copy(pick);
    EXPECT_EQ(copy.getNetwork(), "UU");
    EXPECT_EQ(copy.getStation(), "CWU");
    EXPECT_EQ(copy.getPhase(), Pick::Phase::S);
    EXPECT_NEAR(copy.getTime(), 1349131125.0860946, 1.e-4);
    EXPECT_EQ(*copy.getIdentifier(), 122);
}

TEST(AssociatorsPhaseLink, Arrival)
{
    Pick pick;
    pick.setNetwork("UU");
    pick.setStation("CWU");
    pick.setPhase(Pick::Phase::S);
    pick.setTime(1349131125.0860946);
    pick.setIdentifier(122);

    Arrival arrival(pick);
    arrival.setProbability(0.94);
    EXPECT_EQ(arrival.getNetwork(), "UU");
    EXPECT_EQ(arrival.getStation(), "CWU");
    EXPECT_EQ(arrival.getPhase(), Arrival::Phase::S);
    EXPECT_NEAR(arrival.getTime(), 1349131125.0860946, 1.e-4);
    EXPECT_EQ(*arrival.getIdentifier(), 122);
    EXPECT_TRUE(std::abs(*arrival.getProbability() -  0.94) < 1.e-10);
}

TEST(AssociatorsPhaseLink, UtahInference)
{
    Inference inference(Region::Utah);
    EXPECT_EQ(inference.getNumberOfFeatures(), 5);
    EXPECT_EQ(inference.getSimulationSize(), 1000);

    inference.load(utahModel);
    std::vector<double> X(1000*5, 0.0);
    std::vector<double> Xwork{0.403565,0.384886,0.0,0.0,0.0,
                              0.379272,0.374152,0.0002778629461924235,0.0,0.0,
                              0.400962,0.354218,0.01470469832420349,0.0,0.0,
                              0.427383,0.383652,0.01926674246788025,0.0,0.0,
                              0.457935,0.365838,0.03884924451510111,0.0,0.0,
                              0.391526,0.324157,0.044431016842524214,0.0,0.0,
                              0.450868,0.335365,0.05695272882779439,0.0,0.0,
                              0.339476,0.319236,0.06611587802569072,0.0,0.0,
                              0.495701,0.389886,0.06886587738990783,0.0,0.0,
                              0.384406,0.300186,0.06911587913831076,0.0,0.0,
                              0.446325,0.439806,0.07194157441457112,0.0,0.0,
                              0.407249,0.45774,0.07913171648979186,0.0,0.0,
                              0.507997,0.361957,0.07993676463762919,0.0,0.0,
                              0.288915,0.421232,0.08384335438410441,0.0,0.0,
                              0.515842,0.360667,0.08543010354042054,0.0,0.0,
                              0.446524,0.29894,0.08711087306340536,0.0,0.0,
                              0.453041,0.467774,0.09908669392267863,0.0,0.0,
                              0.510023,0.319689,0.1011992116769155,0.0,0.0,
                              0.539121,0.390023,0.10567780335744222,0.0,0.0,
                              0.34996,0.256574,0.1187949518362681,0.0,0.0,
                              0.425045,0.239718,0.13767133553822836,0.0,0.0,
                              0.427749,0.212291,0.16904423236846924,0.0,0.0,
                              0.193939,0.490133,0.1818803608417511,0.0,0.0,
                              0.493679,0.552162,0.19951353073120118,0.0,0.0};
    std::vector<double> refProbabilities
    {
        0.99900717,0.9669727, 0.9516068, 0.94454986,0.94161916,0.94305223,
        0.9409335, 0.9468487, 0.9495476, 0.94515014,0.95276326,0.958388,
        0.9511834, 0.9508108, 0.94603825,0.93780273,0.9464146, 0.9413374,
        0.931256,  0.92510706,0.92678374,0.9182009, 0.9025983, 0.8686945,
        0.05338312
    };
    std::copy(Xwork.begin(), Xwork.end(), X.begin());
    for (int i = 24; i < 1000; ++i)
    {
        X[5*i + 4] = 1;
    }
                            
    auto result = inference.predictProbability(X);
    for (size_t i = 0; i < refProbabilities.size(); ++i)
    {
        EXPECT_TRUE(std::abs(result[i] - refProbabilities[i]) < 1.e-5);
    }
/*
std::cout << result.size() << std::endl;
for (int i =0 ; i< 9; ++i)
{
  std::cout << result[i] << std::endl;
}
result = inference.predictProbability(24, Xwork);
for (int i = 0; i < 9; ++i)
{
  std::cout << i <<  " "<< result[i] << std::endl;
}
*/

/*
    auto picks = picksFor60000197();
    auto associations = inference.associate(picks, 0.5);
    EXPECT_EQ(associations.size(), 1);
    EXPECT_EQ(associations[0].size(), picks.size());
    for (size_t iPick = 0; iPick < associations[0].size(); ++iPick)
    {
        EXPECT_NEAR(result[iPick], *associations[0][iPick].getProbability(), 1.e-4);
        EXPECT_EQ(picks[iPick].getStation(), associations[0][iPick].getStation()); 
        EXPECT_TRUE( std::abs(picks[iPick].getTime() - associations[0][iPick].getTime() ) < 1.e-4 );
        if (picks[iPick].getPhase() == Pick::Phase::P)
        {
            EXPECT_EQ(associations[0][iPick].getPhase(), Arrival::Phase::P);
        }
        else
        {
            EXPECT_EQ(associations[0][iPick].getPhase(), Arrival::Phase::S);
        }
//        std::cout << result[iPick] << " " << *associations[0][iPick].getProbability() << associations[0][iPick].getStation() << std::endl; 
    }
//getchar();
*/
}

TEST(AssociatorsPhaseLink, InvalidStation)
{
    auto eqPicks = picksFor60000637();
    auto invalidPick = eqPicks.at(0);
    invalidPick.setNetwork("XX");
    invalidPick.setStation("FAKE");
    eqPicks.push_back(invalidPick);

    Inference inference(Region::Utah);
    inference.load(utahModel);
    int minimumClusterSize = 5;
    auto associations = inference.associate(eqPicks, minimumClusterSize, 0.5);
    EXPECT_TRUE(associations.size() == 1);
    if (!associations.empty())
    {
        EXPECT_TRUE(eqPicks.size() == associations[0].size() + 1);
    }
}

TEST(AssociatorsPhaseLink, UtahInferenceChallenge)
{
    Inference inference(Region::Utah);
    inference.load(utahModel);

    auto blastPicks = picksFor60000197();
    auto pickToInsert = blastPicks[4];
    pickToInsert.setTime(blastPicks[0].getTime() - 15);
    pickToInsert.setPhase(Pick::Phase::S);
    blastPicks.push_back(pickToInsert);

    int minimumClusterSize = 5;
    auto associations = inference.associate(blastPicks, minimumClusterSize, 0.5);
    EXPECT_TRUE(associations.size() == 1);
    if (!associations.empty()){EXPECT_TRUE(associations[0].size() == 24);}
 std::cout << associations.size() << std::endl;
 std::cout << associations[0].size() << std::endl;

    // Overlap
    blastPicks = picksFor60000197();
    auto eqPicks = picksFor60000637();

    auto t0blast = blastPicks.at(0).getTime();
    auto t0eq = eqPicks.at(0).getTime();
    auto picks = picksFor60000197();
    for (auto &eqPick : eqPicks)
    {
        auto t = (eqPick.getTime() - t0eq) + t0blast + 10;
        eqPick.setTime(t);
        picks.push_back(eqPick);
    }
    
/*
    auto nPicks = static_cast<int> (picks.size());
    for (int i = 0; i < nPicks; ++i)
    {
        auto pick = picks[i];
        pick.setTime(pick.getTime() + 20);
        picks.push_back(pick);
    }
std::cout << " " << std::endl;
std::cout << picks.size() << std::endl;
*/
    associations = inference.associate(picks, minimumClusterSize, 0.5);
    std::cout << associations.size() << std::endl;
    std::cout << associations[0].size() << std::endl;
    //getchar();
}
