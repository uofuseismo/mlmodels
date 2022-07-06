#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#ifndef NDEBUG
#include <cassert>
#endif
#include "uuss/features/magnitude/threeChannelFeatures.hpp"
#include "uuss/features/magnitude/temporalFeatures.hpp"
#include "uuss/features/magnitude/spectralFeatures.hpp"
#include "uuss/features/magnitude/channelFeatures.hpp"
#include "uuss/features/magnitude/channel.hpp"
#include "uuss/features/magnitude/hypocenter.hpp"

#define TARGET_SAMPLING_RATE 100    // 100 Hz
#define TARGET_SAMPLING_PERIOD 0.01 // 1/100
#define TARGET_SIGNAL_LENGTH 650    // 1.5s before to 5s after
#define PRE_ARRIVAL_TIME 1.5        // 1.5s before S arrival
#define POST_ARRIVAL_TIME 5         // 5s after S arrival
#define S_PICK_ERROR 0.10           // Alysha's S pickers are about twice as noise as the P pick so 0.05 seconds -> 0.1 seconds

using namespace UUSS::Features::Magnitude;

