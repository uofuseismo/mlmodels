#include <string>
#include "uuss/features/version.hpp"

using namespace UUSS::Features;

int Version::getMajor() noexcept
{
    return UUSS_FEATURES_MAJOR;
}

int Version::getMinor() noexcept
{
    return UUSS_FEATURES_MINOR;
}

int Version::getPatch() noexcept
{
    return UUSS_FEATURES_PATCH;
}

bool Version::isAtLeast(const int major, const int minor,
                        const int patch) noexcept
{
    if (UUSS_FEATURES_MAJOR < major){return false;}
    if (UUSS_FEATURES_MAJOR > major){return true;}
    if (UUSS_FEATURES_MINOR < minor){return false;}
    if (UUSS_FEATURES_MINOR > minor){return true;}
    if (UUSS_FEATURES_PATCH < patch){return false;}
    return true;
}

std::string Version::getVersion() noexcept
{
    std::string version(UUSS_FEATURES_VERSION);
    return version;
}
