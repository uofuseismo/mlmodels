#ifndef PRIVATE_SHIFTLONGITUDE
#define PRIVATE_SHIFTLONGITUDE
#include <limits>
#ifndef NDEBUG
#include <cassert>
#endif

namespace
{
[[nodiscard]] double shiftLongitude(const double lonIn)
{
    double lon = lonIn;
    if (lon < -180)
    {
        for (int k = 0; k < std::numeric_limits<int>::max(); ++k)
        {
            double tempLon = lon + k*360;
            if (tempLon >= -180)
            {
                lon = tempLon;
                break;
            }
        }
    }
    if (lon >= 180)
    {
        for (int k = 0; k < std::numeric_limits<int>::max(); ++k)
        {
            double tempLon = lon - k*360;
            if (tempLon < 180)
            {
                lon = tempLon;
                break;
            }
        }
    }
#ifndef NDEBUG
    assert(lon >= -180 && lon < 180);
#endif
    return lon;
}
}

#endif
