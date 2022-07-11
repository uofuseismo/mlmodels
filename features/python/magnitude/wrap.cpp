#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <uuss/features/magnitude/hypocenter.hpp>
#include <uuss/features/magnitude/channel.hpp>
#include <uuss/features/magnitude/pFeatures.hpp>
#include <uuss/features/magnitude/sFeatures.hpp>
#include <uuss/features/magnitude/temporalFeatures.hpp>
#include <uuss/features/magnitude/spectralFeatures.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//#include "wrap.hpp"
#include "initialize.hpp"

namespace
{
struct SimpleResponse
{
    //SimpleResponse(const SimpleResponse &) = default;
    //SimpleResponse(SimpleResponse &&) noexcept = default;
    //SimpleResponse& operator=(const SimpleResponse &) = default;
    //SimpleResponse& operator=(SimpleResponse &&) noexcept = default; 
    void setValue(const double value)
    {
        if (value == 0){throw std::invalid_argument("Value is 0");}
        mValue = value;
    }
    double getValue() const{return mValue;}
    void setUnits(const std::string &units){mUnits = units;}
    std::string getUnits() const{return mUnits;}
    std::string mUnits;
    double mValue{0};
};
    
class Hypocenter
{
public:
    /// C'tor
    Hypocenter() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::Hypocenter> ())
    {
    }
    Hypocenter(const Hypocenter &hypo){*this = hypo;}
    Hypocenter(const UUSS::Features::Magnitude::Hypocenter &hypo)
    {
        *this = hypo;
    }
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
    Hypocenter& operator=(const UUSS::Features::Magnitude::Hypocenter &hypo)
    {
        pImpl = std::make_unique<UUSS::Features::Magnitude::Hypocenter> (hypo);
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

//----------------------------------------------------------------------------//

class Channel
{
public:
    /// C'tor
    Channel() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::Channel> ()) 
    {   
    }   
    Channel(const Channel &channel){*this = channel;}
    Channel(Channel &&channel) noexcept{*this = std::move(channel);}
    /// Destructor
    ~Channel() = default;
    /// Copy assignment
    Channel& operator=(const Channel &channel)
    {
        if (&channel == this){return *this;}
        pImpl = std::make_unique<UUSS::Features::Magnitude::Channel>
                (*channel.pImpl);
        return *this;
    }   
    /// Move assignment
    Channel& operator=(Channel &&channel) noexcept
    {
        if (&channel == this){return *this;}
        pImpl = std::move(channel.pImpl);
        return *this;
    }   
    /// Latitude
    void setLatitude(const double latitude){pImpl->setLatitude(latitude);}
    double getLatitude() const{return pImpl->getLatitude();}
    /// Longitude
    void setLongitude(const double longitude){pImpl->setLongitude(longitude);}
    double getLongitude() const{return pImpl->getLongitude();}
    /// Sampling rate
    void setSamplingRate(const double samplingRate)
    {
        pImpl->setSamplingRate(samplingRate);
    }
    double getSamplingRate() const{return pImpl->getSamplingRate();}
    /// Simple resopnse
    void setSimpleResponse(const SimpleResponse &response)
    {
        pImpl->setSimpleResponse(response.getValue(),
                                 response.getUnits());
    } 
    SimpleResponse getSimpleResponse() const
    {
        SimpleResponse response;
        response.setValue(pImpl->getSimpleResponseValue());
        response.setUnits(pImpl->getSimpleResponseUnits());
        return response;
    }
    void setAzimuth(const double azimuth){pImpl->setAzimuth(azimuth);}
    double getAzimuth() const{return pImpl->getAzimuth();}
    /// Network
    void setNetworkCode(const std::string &network)
    {
        pImpl->setNetworkCode(network);
    }
    std::string getNetworkCode() const{return pImpl->getNetworkCode();}
    void setStationCode(const std::string &station)
    {
        pImpl->setStationCode(station);
    }
    std::string getStationCode() const{return pImpl->getStationCode();}
    void setChannelCode(const std::string &channel)
    {
        pImpl->setChannelCode(channel);
    }
    std::string getChannelCode() const{return pImpl->getChannelCode();}
    void setLocationCode(const std::string &locationCode)
    {
        pImpl->setLocationCode(locationCode);
    }
    std::string getLocationCode() const{return pImpl->getLocationCode();}
    void clear() noexcept{pImpl->clear();}
        

    std::unique_ptr<UUSS::Features::Magnitude::Channel> pImpl;
};

//----------------------------------------------------------------------------//
class SpectralFeatures
{
public:
    SpectralFeatures() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::SpectralFeatures> ()) 
    {   
    }   
    SpectralFeatures(const SpectralFeatures &features)
    {
        *this = features;
    }   
    SpectralFeatures(const UUSS::Features::Magnitude::SpectralFeatures &features)
    {
        *this = features;
    }   
    SpectralFeatures(SpectralFeatures &&features) noexcept
    {
        *this = std::move(features);
    }   
    SpectralFeatures& operator=(const UUSS::Features::Magnitude::SpectralFeatures &features)
    {   
        pImpl = std::make_unique<UUSS::Features::Magnitude::SpectralFeatures> (features);
        return *this;
    }
    SpectralFeatures& operator=(const SpectralFeatures &features)
    {
        if (&features == this){return *this;}
        pImpl = std::make_unique<UUSS::Features::Magnitude::SpectralFeatures>
                (*features.pImpl);
        return *this;
    }
    SpectralFeatures& operator=(SpectralFeatures &&features) noexcept
    {   
        if (&features == this){return *this;}
        pImpl = std::move(features.pImpl);
        return *this;
    }
    ~SpectralFeatures() = default;
    std::pair<double, double> getDominantFrequencyAndAmplitude() const
    {
        return pImpl->getDominantFrequencyAndAmplitude();
    }
    std::pair<std::vector<double>, std::vector<double>> getAverageFrequenciesAndAmplitudes() const
    {
        auto avg = pImpl->getAverageFrequenciesAndAmplitudes();
        std::vector<double> freqs(avg.size());
        std::vector<double> amps(avg.size());
        for (int i = 0; i < static_cast<int> (amps.size()); ++i)
        {
            freqs[i] = avg[i].first;
            amps[i]  = avg[i].second;
        }
        return std::pair(freqs, amps);
    }
    std::unique_ptr<UUSS::Features::Magnitude::SpectralFeatures> pImpl;
};

//----------------------------------------------------------------------------//

class TemporalFeatures
{
public:
    TemporalFeatures() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::TemporalFeatures> ())
    {
    }
    TemporalFeatures(const TemporalFeatures &features)
    {
        *this = features;
    }
    TemporalFeatures(const UUSS::Features::Magnitude::TemporalFeatures &features)
    {
        *this = features;
    }
    TemporalFeatures(TemporalFeatures &&features) noexcept
    {
        *this = std::move(features);
    }
    TemporalFeatures& operator=(const UUSS::Features::Magnitude::TemporalFeatures &features)
    {
        pImpl = std::make_unique<UUSS::Features::Magnitude::TemporalFeatures> (features);
        return *this;
    }
    TemporalFeatures& operator=(const TemporalFeatures &features)
    {
        if (&features == this){return *this;}
        pImpl = std::make_unique<UUSS::Features::Magnitude::TemporalFeatures>
                (*features.pImpl);
        return *this;
    }
    TemporalFeatures& operator=(TemporalFeatures &&features) noexcept
    {
        if (&features == this){return *this;}
        pImpl = std::move(features.pImpl);
        return *this;
    }
    ~TemporalFeatures() = default;
    void setVariance(const double variance)
    {
        if (variance < 0)
        {
            throw std::invalid_argument("Variance must be positive");
        }
        pImpl->setVariance(variance);
    }
    double getVariance() const
    {
        return pImpl->getVariance();
    }
    void setMinimumAndMaximumValue(const std::pair<double, double> &minMax)
    {
        pImpl->setMinimumAndMaximumValue(minMax);
    }
    std::pair<double, double> getMinimumAndMaximumValue() const
    {
        return pImpl->getMinimumAndMaximumValue();
    }
    void clear() noexcept{pImpl->clear();}
    std::unique_ptr<UUSS::Features::Magnitude::TemporalFeatures> pImpl;
};

//----------------------------------------------------------------------------//

class PFeatures
{
public:
    PFeatures() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::PFeatures> ())
    {
    }
    ~PFeatures() = default;

    void initialize(const ::Channel &channel)
    {
        pImpl->initialize(*channel.pImpl);
    }
    bool isInitialized() const noexcept
    {
        return pImpl->isInitialized();
    } 
    void setHypocenter(const ::Hypocenter &hypo)
    {
        pImpl->setHypocenter(*hypo.pImpl);
    }
    ::Hypocenter getHypocenter() const
    {
        Hypocenter hypo(pImpl->getHypocenter());
        return hypo;
    }
    void process(const pybind11::array_t<double, pybind11::array::c_style |
                                                 pybind11::array::forcecast> &array,
                 const double arrivalTime)
    {
        std::vector<double> x(array.size());
        std::copy(array.data(), array.data() + array.size(), x.begin());
        pImpl->process(x, arrivalTime);
    }
    double getSourceReceiverDistance() const
    {
        return pImpl->getSourceReceiverDistance();
    }
    double getBackAzimuth() const
    {
        return pImpl->getBackAzimuth();
    }
    ::TemporalFeatures getTemporalNoiseFeatures() const
    {
        auto features = pImpl->getTemporalNoiseFeatures();
        return TemporalFeatures(features);
    }
    ::TemporalFeatures getTemporalSignalFeatures() const
    {
        auto features = pImpl->getTemporalSignalFeatures();
        return TemporalFeatures(features);
    }
    ::SpectralFeatures getSpectralNoiseFeatures() const
    {   
        auto features = pImpl->getSpectralNoiseFeatures();
        return SpectralFeatures(features);
    }   
    ::SpectralFeatures getSpectralSignalFeatures() const
    {   
        auto features = pImpl->getSpectralSignalFeatures();
        return SpectralFeatures(features);
    }
    pybind11::array_t<double> getVelocitySignal() const
    {
        auto signal = pImpl->getVelocitySignal();
        auto y = pybind11::array_t<double, pybind11::array::c_style>
                 (signal.size()); 
        pybind11::buffer_info yBuffer = y.request();
        auto yPtr = static_cast<double *> (yBuffer.ptr);
        std::copy(signal.begin(), signal.end(), yPtr);
        return y;
    }
    void clear() noexcept{pImpl->clear();}
    std::unique_ptr<UUSS::Features::Magnitude::PFeatures> pImpl;
    PFeatures(const PFeatures &) = delete;
    PFeatures(PFeatures &&) noexcept = delete;
    PFeatures& operator=(const PFeatures &) = delete;
    PFeatures& operator=(PFeatures &&) noexcept = delete;
};

class SFeatures
{
public:
    SFeatures() :
        pImpl(std::make_unique<UUSS::Features::Magnitude::SFeatures> ())
    {
    }
    ~SFeatures() = default;
    void initialize(const ::Channel &nChannel,
                    const ::Channel &eChannel)
    {
        pImpl->initialize(*nChannel.pImpl,
                          *eChannel.pImpl);
    }
    bool isInitialized() const noexcept
    {
        return pImpl->isInitialized();
    }
    void setHypocenter(const ::Hypocenter &hypo)
    {
        pImpl->setHypocenter(*hypo.pImpl);
    }
    ::Hypocenter getHypocenter() const
    {
        Hypocenter hypo(pImpl->getHypocenter());
        return hypo;
    }
    void process(const pybind11::array_t<double, pybind11::array::c_style |
                                                 pybind11::array::forcecast> &nArray,
                 const pybind11::array_t<double, pybind11::array::c_style |
                                                 pybind11::array::forcecast> &eArray,
                 const double arrivalTime)
    {
        if (!isInitialized()){throw std::runtime_error("Class not initialized");}
        if (nArray.size() != eArray.size())
        {
            throw std::runtime_error("n_signal.size() ! = e_signal.size()");
        }
        std::vector<double> north(nArray.size());
        std::copy(nArray.data(), nArray.data() + nArray.size(), north.begin());

        std::vector<double> east(eArray.size());
        std::copy(eArray.data(), eArray.data() + eArray.size(), east.begin());

        pImpl->process(north, east, arrivalTime);
    }
    pybind11::array_t<double> getRadialVelocitySignal() const
    {
        auto signal = pImpl->getRadialVelocitySignal();
        auto y = pybind11::array_t<double, pybind11::array::c_style>
                 (signal.size());
        pybind11::buffer_info yBuffer = y.request();
        auto yPtr = static_cast<double *> (yBuffer.ptr);
        std::copy(signal.begin(), signal.end(), yPtr);
        return y;
    }
    pybind11::array_t<double> getTransverseVelocitySignal() const
    {
        auto signal = pImpl->getTransverseVelocitySignal();
        auto y = pybind11::array_t<double, pybind11::array::c_style>
                 (signal.size());
        pybind11::buffer_info yBuffer = y.request();
        auto yPtr = static_cast<double *> (yBuffer.ptr);
        std::copy(signal.begin(), signal.end(), yPtr);
        return y;
    }
    double getSourceReceiverDistance() const
    {   
        return pImpl->getSourceReceiverDistance();
    }   
    double getBackAzimuth() const
    {   
        return pImpl->getBackAzimuth();
    }
    ::TemporalFeatures getRadialTemporalNoiseFeatures() const
    {
        auto features = pImpl->getRadialTemporalNoiseFeatures();
        return TemporalFeatures(features);
    }
    ::TemporalFeatures getRadialTemporalSignalFeatures() const
    {   
        auto features = pImpl->getRadialTemporalSignalFeatures();
        return TemporalFeatures(features);
    }   
    ::SpectralFeatures getRadialSpectralNoiseFeatures() const
    {   
        auto features = pImpl->getRadialSpectralNoiseFeatures();
        return SpectralFeatures(features);
    }   
    ::SpectralFeatures getRadialSpectralSignalFeatures() const
    {   
        auto features = pImpl->getRadialSpectralSignalFeatures();
        return SpectralFeatures(features);
    }
    ::TemporalFeatures getTransverseTemporalNoiseFeatures() const
    {   
        auto features = pImpl->getTransverseTemporalNoiseFeatures();
        return TemporalFeatures(features);
    }   
    ::TemporalFeatures getTransverseTemporalSignalFeatures() const
    {   
        auto features = pImpl->getTransverseTemporalSignalFeatures();
        return TemporalFeatures(features);
    }   
    ::SpectralFeatures getTransverseSpectralNoiseFeatures() const
    {   
        auto features = pImpl->getTransverseSpectralNoiseFeatures();
        return SpectralFeatures(features);
    }   
    ::SpectralFeatures getTransverseSpectralSignalFeatures() const
    {   
        auto features = pImpl->getTransverseSpectralSignalFeatures();
        return SpectralFeatures(features);
    }
    void clear() noexcept{pImpl->clear();}
    std::unique_ptr<UUSS::Features::Magnitude::SFeatures> pImpl;
    SFeatures(const SFeatures &) = delete;
    SFeatures(SFeatures &&) noexcept = delete;
    SFeatures& operator=(const SFeatures &) = delete;
    SFeatures& operator=(SFeatures &&) noexcept = delete;
};

}

///---------------------------------------------------------------------------//
//                                Initialization                              //
//----------------------------------------------------------------------------//
void PyFeatures::Magnitude::initialize(pybind11::module &m)
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
    // Copy rules
    /*
    h.def("__copy__", [](const ::Hypocenter &self)
          {
             return ::Hypocenter(self);
          });
    h.def("__deepcopy__", [](const ::Hypocenter &self)
          {
             return ::Hypocenter(self);
          });
    */
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
    //------------------------------------------------------------------------//
    pybind11::class_<::SimpleResponse> sr(m, "SimpleResponse");
    sr.def(pybind11::init<> ());
    sr.doc() = R"""(
Defines a simple response.

Properties
----------
   units : str
       The units which can be DU/M/S or DU/M/S**2
   value : float
       The value which after dividing the signal by will result in units of M/S or M/S**2.
)""";
    sr.def_property("value",
                    &::SimpleResponse::getValue, &::SimpleResponse::setValue);
    sr.def_property("units",
                    &::SimpleResponse::getUnits, &::SimpleResponse::setUnits);
    // Copy rules
    /*
    sr.def("__copy__", [](const ::SimpleResponse &self)
           {
              return ::SimpleResponse(self);
           });
    sr.def("__deepcopy__", [](const ::SimpleResponse &self)
           {
              return ::SimpleResponse(self);
           });
    */
    // Pickling rules
    sr.def(pybind11::pickle(
        [](const ::SimpleResponse &response)
        {
            auto units = response.getUnits();
            auto value = response.getValue();
            return pybind11::make_tuple(units, value);
        },
        [](pybind11::tuple t)
        {
            if (t.size() != 2){throw std::runtime_error("Invalid state");}
            auto units = t[0].cast<std::string> (); 
            auto value = t[1].cast<double> (); 
            ::SimpleResponse response;
            response.setUnits(units);
            response.setValue(value);
            return response;
        }
    )); 

    //------------------------------------------------------------------------//

    pybind11::class_<::Channel> c(m, "Channel"); 
    c.def(pybind11::init<> ());
    c.doc() = R"""(
Defines a channel.

Required Properties
-------------------

   sampling_rate : float
              The channel's nominal sampling rate in Hz.
   simple_response : SimpleResponse
              The value of the simple response and the AQMS-based string defining
              the simple response which can be DU/M/S and DU/M/S**2.
              When the signal is divided by this value then
              the result will be in the units of M/S or M/S**2.
   latitude : float
              The station's latitude in degrees.  This must be in the range [-90,90].
   longitude : float
              The station's longitude in degrees where positive increases east.

Optional Properties
-------------------
   azimuth : float
              The channel's azimuth measured positive east of north in degrees.
              For example, this will be 0 for north and vertical channels
              and will be 90 for east channels.  This must be in the range [0,360).
   network_code : str
              The network code - e.g., UU.
   station_code : str
              The station code - e.g., SRU.
   channel_code : str
              The channel code - e.g., HHZ for a high-sample rate broadband 
              vertical channel that is likely sensitive to ground velocity or
              ENZ for a high-sample rate strong-motion vertical channel that is
              likely sensitive to ground acceleration.
   location_code : str
              The location code - e.g., 01.

)""";
    c.def_property("sampling_rate",
                   &::Channel::getSamplingRate, &::Channel::setSamplingRate);
    c.def_property("simple_response",
                   &::Channel::getSimpleResponse, &::Channel::setSimpleResponse);
    c.def_property("latitude",
                   &::Channel::getLatitude, &::Channel::setLatitude);
    c.def_property("longitude",
                   &::Channel::getLongitude, &::Channel::setLongitude);
    c.def_property("azimuth",
                   &::Channel::getAzimuth, &::Channel::setAzimuth);
    c.def_property("network_code",
                   &::Channel::getNetworkCode, &::Channel::setNetworkCode);
    c.def_property("station_code",
                   &::Channel::getStationCode, &::Channel::setStationCode);
    c.def_property("channel_code",
                   &::Channel::getChannelCode, &::Channel::setChannelCode);
    c.def_property("location_code",
                   &::Channel::getLocationCode, &::Channel::setLocationCode);
    c.def("clear", &::Channel::clear, "Resets the class.");
    // Copy rules
     /*
    c.def("__copy__", [](const ::Channel &self)
          {
             return ::Channel(self);
          });
    c.def("__deepcopy__", [](const ::Channel &self)
          {
             return ::Channel(self);
          });
    */
    // Pickling rules
    c.def(pybind11::pickle(
        [](const ::Channel &channel)
        {
            auto samplingRate = channel.getSamplingRate();
            auto [gain, units] = channel.getSimpleResponse();
            auto latitude = channel.getLatitude();
            auto longitude = channel.getLongitude();
            double azimuth =-1000;
            try
            {
                azimuth = channel.getAzimuth();
            }
            catch (...)
            {
            }
            auto networkCode = channel.getNetworkCode();
            auto stationCode = channel.getStationCode();
            auto channelCode = channel.getChannelCode();
            auto locationCode = channel.getLocationCode();
            return pybind11::make_tuple(networkCode, stationCode, channelCode, locationCode,
                                        units, gain, 
                                        samplingRate,
                                        latitude, longitude, azimuth);
        },
        [](pybind11::tuple t)
        {
            if (t.size() != 10){throw std::runtime_error("Invalid state");}
            auto networkCode  = t[0].cast<std::string> ();
            auto stationCode  = t[1].cast<std::string> ();
            auto channelCode  = t[2].cast<std::string> ();
            auto locationCode = t[3].cast<std::string> ();
            auto units        = t[4].cast<std::string> ();
            auto value        = t[5].cast<double> ();
            auto samplingRate = t[6].cast<double> ();
            auto latitude     = t[7].cast<double> ();
            auto longitude    = t[8].cast<double> ();
            auto azimuth      = t[9].cast<double> ();
            ::Channel channel;
            if (!networkCode.empty()){channel.setNetworkCode(networkCode);}
            if (!stationCode.empty()){channel.setStationCode(stationCode);}
            if (!channelCode.empty()){channel.setChannelCode(channelCode);}
            if (!locationCode.empty()){channel.setLocationCode(locationCode);}
            channel.setSamplingRate(samplingRate);
            SimpleResponse response;
            response.setValue(value);
            response.setUnits(units); 
            channel.setSimpleResponse(response);
            channel.setLatitude(latitude);
            channel.setLongitude(longitude);
            if (azimuth >= 0 && azimuth < 360){channel.setAzimuth(azimuth);}
            return channel;
        }
    ));
 
    //------------------------------------------------------------------------//

    pybind11::class_<::TemporalFeatures> tf(m, "TemporalFeatures");
    tf.def(pybind11::init<> ());
    tf.doc() = R"""(
The time-based features.

Properties
----------
     minimum_and_maximum_value : List[float, float]
        The minimum and maximum amplitude.  This is units of micrometers/second.
     variance : float
        The variance.  This is the signal power minus the DC power.  This is
        in units of (micrometers/second)^2
)""";
    tf.def_property("variance",
                    &::TemporalFeatures::getVariance, &::TemporalFeatures::setVariance);
    tf.def_property("minimum_and_maximum_value",
                    &::TemporalFeatures::getMinimumAndMaximumValue,
                    &::TemporalFeatures::setMinimumAndMaximumValue);
    tf.def("clear", &::TemporalFeatures::clear, "Resets the class.");
    tf.def("__repr__",
           [](const ::TemporalFeatures &f)
           {
               std::stringstream s;
               s << *f.pImpl;
               return s.str();
           });

    pybind11::class_<::SpectralFeatures> sf(m, "SpectralFeatures");
    sf.def(pybind11::init<> ());
    sf.doc() = R"""(
The spectral-based features.

Read-Only Properties
--------------------
   dominant_frequency_and_amplitude : List[float, float]
       The dominant frequency in Hz and the corresponding largest amplitude.
   average_amplitudes : List[array, array]
       The average amplitude in the computation window at the given frequencies.
)""";
    sf.def_property_readonly("dominant_frequency_and_amplitude",
                             &::SpectralFeatures::getDominantFrequencyAndAmplitude);
    sf.def_property_readonly("average_frequencies_and_amplitudes",
                             &::SpectralFeatures::getAverageFrequenciesAndAmplitudes);
   
    //------------------------------------------------------------------------//

    pybind11::class_<::PFeatures> pfeatures(m, "PFeatures"); 
    pfeatures.def(pybind11::init<> ());
    pfeatures.doc() = R"""(
Extracts the features P-arrival features from the vertical channel.

Properties
----------
    hypocenter : Hypocenter
       The hypocentral information.

Read-only Properties
--------------------
   velocity_signal : np.array
       The processed velocity signal.  This should be in units of
       micrometers/second.
   source_receiver_distance : float
       The source-receiver distance in km. 
   back_azimuth : float
       The receiver-to-source azimuth in degrees measured positive east of north.
   spectral_noise_features : SpectralFeatures
       The spectral features of the noise.
   spectral_signal_features : SpectralFeatures
       The spectral features of the signal.
   temporal_signal_features : TemporalFeatures
       The temporal features of the noise.
   temporal_signal_features : TemporalFeatures
       the temporal features of the signal. 
)""";
    pfeatures.def("initialize", &::PFeatures::initialize,
                  "Initializes the feature extractor based on the channel information.");
    pfeatures.def_property("hypocenter",
                           &::PFeatures::getHypocenter,
                           &::PFeatures::setHypocenter);
    pfeatures.def("process", &::PFeatures::process,
                  "Processes the waveform.  Additionally, the arrival time relative to the window start must be specified.");
    pfeatures.def_property_readonly("velocity_signal",
                                    &::PFeatures::getVelocitySignal);
    pfeatures.def_property_readonly("source_receiver_distance",
                                    &::PFeatures::getSourceReceiverDistance);
    pfeatures.def_property_readonly("back_azimuth",
                                    &::PFeatures::getBackAzimuth);
    pfeatures.def_property_readonly("spectral_noise_features",
                                    &::PFeatures::getSpectralNoiseFeatures);
    pfeatures.def_property_readonly("spectral_signal_features",
                                    &::PFeatures::getSpectralSignalFeatures);
    pfeatures.def_property_readonly("temporal_noise_features",
                                    &::PFeatures::getTemporalNoiseFeatures);
    pfeatures.def_property_readonly("temporal_signal_features",
                                    &::PFeatures::getTemporalSignalFeatures);

    //------------------------------------------------------------------------//

    pybind11::class_<::SFeatures> sfeatures(m, "SFeatures"); 
    sfeatures.def(pybind11::init<> ());
    sfeatures.doc() = R"""(
Extracts the features SH and SV-arrival features from a three-component seismogram.

Properties
----------
    hypocenter : Hypocenter
       The hypocentral information.

Read-only Properties
--------------------
   radial_velocity_signal : np.array
       The processed radial velocity signal.  This should be in units of
       micrometers/second.
   source_receiver_distance : float
       The source-receiver distance in km. 
   back_azimuth : float
       The receiver-to-source azimuth in degrees measured positive east of north.
   spectral_noise_features : SpectralFeatures
       The spectral features of the noise.
   spectral_signal_features : SpectralFeatures
       The spectral features of the signal.
   temporal_signal_features : TemporalFeatures
       The temporal features of the noise.
   temporal_signal_features : TemporalFeatures
       the temporal features of the signal. 
)""";
    sfeatures.def("initialize",
                  &::SFeatures::initialize,
                  "Initializes the feature extractor based on the north (1) and east (2) channel information.");
    sfeatures.def_property("hypocenter",
                           &::SFeatures::getHypocenter,
                           &::SFeatures::setHypocenter);
    sfeatures.def("process", &::SFeatures::process,
                  "Processes the north and east waveforms.  Additionally, the arrival time relative to the window start must be specified.");
    sfeatures.def_property_readonly("radial_velocity_signal",
                                    &::SFeatures::getRadialVelocitySignal);
    sfeatures.def_property_readonly("transverse_velocity_signal",
                                    &::SFeatures::getTransverseVelocitySignal);
    sfeatures.def_property_readonly("source_receiver_distance",
                                    &::SFeatures::getSourceReceiverDistance);
    sfeatures.def_property_readonly("back_azimuth",
                                    &::SFeatures::getBackAzimuth);
    sfeatures.def_property_readonly("radial_spectral_noise_features",
                                    &::SFeatures::getRadialSpectralNoiseFeatures);
/*
    sfeatures.def_property_readonly("radial_spectral_signal_features",
                                    &::SFeatures::getRadialSpectralSignalFeatures);
    sfeatures.def_property_readonly("radial_temporal_noise_features",
                                    &::SFeatures::getRadialTemporalNoiseFeatures);
    sfeatures.def_property_readonly("radial_temporal_signal_features",
                                    &::SFeatures::getRadialTemporalSignalFeatures);
    sfeatures.def_property_readonly("transverse_spectral_noise_features",
                                    &::SFeatures::getTransverseSpectralNoiseFeatures);
    sfeatures.def_property_readonly("transverse_spectral_signal_features",
                                    &::SFeatures::getTransverseSpectralSignalFeatures);
    sfeatures.def_property_readonly("transverse_temporal_noise_features",
                                    &::SFeatures::getTransverseTemporalNoiseFeatures);
    sfeatures.def_property_readonly("transverse_temporal_signal_features",
                                    &::SFeatures::getTransverseTemporalSignalFeatures);
*/
}
