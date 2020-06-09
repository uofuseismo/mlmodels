#include <string>
#include "uuss/version.hpp"

using namespace UUSS;

int Version::getMajor() noexcept
{
    return UUSSMLMODELS_MAJOR;
}

int Version::getMinor() noexcept
{
    return UUSSMLMODELS_MINOR;
}

int Version::getPatch() noexcept
{
    return UUSSMLMODELS_PATCH;
}

bool Version::isAtLeast(const int major, const int minor,
                        const int patch) noexcept
{
    if (UUSSMLMODELS_MAJOR < major){return false;}
    if (UUSSMLMODELS_MAJOR > major){return true;}
    if (UUSSMLMODELS_MINOR < minor){return false;}
    if (UUSSMLMODELS_MINOR > minor){return true;}
    if (UUSSMLMODELS_PATCH < patch){return false;}
    return true;
}

std::string Version::getVersion() noexcept
{
    std::string version(std::to_string(getMajor()) + "."
                      + std::to_string(getMinor()) + "."
                      + std::to_string(getPatch()));
    return version;
}
