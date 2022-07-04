#ifndef PRIVATE_DISTANCEAZIMUTH_HPP
#define PRIVATE_DISTANCEAZIMUTH_HPP
#include <GeographicLib/Geodesic.hpp>
#include <GeographicLib/Constants.hpp>
#include "shiftLongitude.hpp"
namespace
{
/// @brief Computes the distance and azimuth between two points on a WGS84
///        reference ellipsoid.
/// @param[in] sourceLatitude     The source latitude in degrees.
/// @param[in] sourceLongitude    The source longitude in degrees.
/// @param[in] receiverLatitude   The receiver latitude in degrees.
/// @param[in] receiverLongitude  The receiver longitude in degrees.
/// @param[out] greatCircleDistance  The great circle distance in degrees
///                                  between the source and receiver.
/// @param[out] distance             The source-receiver distance in km.
/// @param[out] azimuth              The source-to-receiver azimuth in degrees.
///                                  This is measured positive clockwise east
///                                  of north.
/// @param[out] backAzimuth          The receiver-to-source azimuth in degrees.
///                                  This is measured postivie clockwise east
///                                  of north.
void computeDistanceAzimuthWGS84(const double sourceLatitude,
                                 const double sourceLongitudeIn,
                                 const double receiverLatitude,
                                 const double receiverLongitudeIn,
                                 double *greatCircleDistance,
                                 double *distance,
                                 double *azimuth,
                                 double *backAzimuth)
{
    if (sourceLatitude < -90 || sourceLatitude > 90)
    {
        throw std::invalid_argument(
            "Source lattiude must be in range [-90,90]");
    }
    if (receiverLatitude < -90 || receiverLatitude > 90)
    {
        throw std::invalid_argument(
            "Receiver latitude must be in range [-90,90]");
    }
    auto sourceLongitude = shiftLongitude(sourceLongitudeIn);
    auto receiverLongitude = shiftLongitude(receiverLongitudeIn);
    // Do calculation
    GeographicLib::Geodesic geodesic{GeographicLib::Constants::WGS84_a(),
                                     GeographicLib::Constants::WGS84_f()};
    *greatCircleDistance
        = geodesic.Inverse(sourceLatitude, sourceLongitude,
                           receiverLatitude, receiverLongitude,
                           *distance, *azimuth, *backAzimuth);
    // Translate azimuth from [-180,180] to [0,360].
    if (*azimuth < 0){*azimuth = *azimuth + 360;}
    // Translate azimuth from [-180,180] to [0,360] then convert to a
    // back-azimuth by subtracting 180, i.e., +180.
    *backAzimuth = *backAzimuth + 180;
}
}
#endif
