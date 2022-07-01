#include <memory>
#include <uuss/features/magnitude/hypocenter.hpp>
//#include "wrap.hpp"
#include "initialize.hpp"

using namespace PFeatures::Magnitude;

namespace
{
class Hypocenter
{
public:
    /// C'tor
    Hypocenter() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::Hypocenter> ())
    {
    }
    Hypocenter(const Hypocenter &hypo){*this = hypo;}
    Hypocenter(Hypocenter &&hypo) noexcept{*this = std::move(hypo);}
    /// Destructor
    ~Hypocenter() = default;
    /// Copy assignment
    Hypocenter& operator=(const Hypocenter &hypo)
    {
        if (&hypo == this){return *this;}
        pImpl = std::make_unique<UUSS::Features::Magnitude::Hypocenter>
                (*hypo.pImpl);
        return *this;
    }
    /// Move assignment
    Hypocenter& operator=(Hypocenter &&hypo) noexcept
    {
        if (&hypo == this){return *this;}
        pImpl = std::move(hypo.pImpl);
        return *this;
    }
    /// Latitude
    void setLatitude(const double latitude){pImpl->setLatitude(latitude);}
    double getLatitude() const{return pImpl->getLatitude();}
    /// Longitude
    void setLongitude(const double longitude){pImpl->setLongitude(longitude);}
    double getLongitude() const{return pImpl->getLongitude();}
    /// Depth
    void setDepth(const double depth){pImpl->setDepth(depth);}
    double getDepth() const{return pImpl->getDepth();}
    /// Evid
    void setEventIdentifier(const int64_t evid)
    {
        pImpl->setEventIdentifier(evid);
    }
    int64_t getEventIdentifier() const{return pImpl->getEventIdentifier();}
    void clear() noexcept{pImpl->clear();}
    

    std::unique_ptr<UUSS::Features::Magnitude::Hypocenter> pImpl;
};
}

///---------------------------------------------------------------------------//
//                                Initialization                              //
//----------------------------------------------------------------------------//
void PFeatures::Magnitude::initialize(pybind11::module &m)
{
    pybind11::class_<::Hypocenter> h(m, "Hypocenter"); 
    h.def(pybind11::init<> ());
    h.doc() = R"""(
Defines a hypocenter.  

Properties
----------

   latitude : float
              The event's latitude in degrees.  This must be in the range [-90,90].
   longitude : float
              The event's longitude in degrees where positive increases east.
   depth : float
           The event's depth in kilometers.
   identifier : int
              The event's identifier.  This is optional.

)""";
    h.def_property("latitude",
                   &::Hypocenter::getLatitude, &::Hypocenter::setLatitude);
    h.def_property("longitude",
                   &::Hypocenter::getLongitude, &::Hypocenter::setLongitude);
    h.def_property("depth",
                   &::Hypocenter::getDepth, &::Hypocenter::setDepth);
    h.def_property("identifier",
                   &::Hypocenter::getEventIdentifier,
                   &::Hypocenter::setEventIdentifier);
    h.def("clear", &::Hypocenter::clear, "Resets the class.");
    // Pickling rules
    h.def(pybind11::pickle(
        [](const ::Hypocenter &hypo)
        {
            auto latitude = hypo.getLatitude();
            auto longitude = hypo.getLongitude();
            auto depth = hypo.getDepth();
            auto evid = hypo.getEventIdentifier(); 
            return pybind11::make_tuple(latitude, longitude, depth, evid);
        },
        [](pybind11::tuple t)
        { 
            if (t.size() != 4){throw std::runtime_error("Invalid state");}
            auto latitude   = t[0].cast<double> ();
            auto longitude  = t[1].cast<double> ();
            auto depth      = t[2].cast<double> ();
            auto identifier = t[3].cast<int64_t> ();
            ::Hypocenter hypo;
            hypo.setLatitude(latitude);
            hypo.setLongitude(longitude);
            hypo.setDepth(depth);
            hypo.setEventIdentifier(identifier);
            return hypo;
        }
    ));
 
}
