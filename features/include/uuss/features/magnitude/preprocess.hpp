#ifndef UUSS_FEATURES_MAGNITUDE_PREPROCESS_HPP
#define UUSS_FEATURES_MAGNITUDE_PREPROCESS_HPP
#include <memory>
namespace UUSS::Features::Magnitude
{
/// @brief This class performs the pre-processing on input signals that are
///        proportional to velocity or acceleration and returns signals velocity
///        signals with units of micrometers/second.
class Preprocess
{
public:
    /// @brief Constructor.
    Preprocess();

    /// @result The target sampling rate in Hz.
    [[nodiscard]] static double getTargetSamplingRate() noexcept;
    /// @result The target sampling period in seconds.
    [[nodiscard]] static double getTargetSamplingPeriod() noexcept;


    /// @brief Initializes the preprocessing.
    /// @param[in] samplingRate    The sampling rate in Hz of the input signal. 
    /// @param[in] simpleResponse  After dividing the signal by this value the
    ///                            signal units will be M/S or M/S**2.
    /// @param[in] units           The units on the simple response.  This can
    ///                            be DU/M/S for velocity or DU/M/S**2 for
    ///                            acceleration.
    /// @param[in] cutStartEnd     cutStartEnd.first is the time relative to
    ///                            the arrival to start the cut window and
    ///                            cutStartEnd.second is the time relative to
    ///                            the arrival to end the cut window.
    ///                            Specifically, the cut start time is:
    ///                               (relative)pickTime + cutStartEnd.first
    ///                            and the cut end time is
    ///                               (relative)pickTime + cutStartEnd.second.
    void initialize(double samplingRate,
                    double simpleResponse,
                    const std::string &units,
                    const std::pair<double, double> &cutStartEnd);
    /// @result The length of the target signal after cutting.
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] int getTargetSignalLength() const;
    /// @result The number of seconds before and after the arrival where the 
    ///         processing will be performed.  For example,
    ///         arrivalTime + result.first will indicate where the processing
    ///         window begins while arrivalTime + result.second will indicate
    ///         where the processing will end.
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] std::pair<double, double> getArrivalTimeProcessingWindow() const;
    /// @result The sampling rate of the input signal in Hz.
    /// @throws std::runtime_error if \c isInitialized() is false.
    [[nodiscard]] double getSamplingRate() const;
    /// @result True indicates the class is initialized.
    [[nodiscard]] bool isInitialized() const noexcept;
 
    /// @brief Processes the signal.
    /// @param[in] signal                      The signal to process.  This
    ///                                        signal has sampling rate 
    ///                                        \c getSamplingRate().
    /// @param[in] arrivalTimeRelativeToStart  The arrival time relative to the
    ///                                        first sample in seconds about 
    ///                                        which to cut the signal.
    /// @throws std::invalid_argument if the signal is too small or the
    ///         arrival time is negative or exceeds the cut window end.
    /// @throws std::runtime_error if \c isInitialized() is false.
    void process(const std::vector<double> &signal,
                 double arrivalTimeRelativeToStart);
    void process(int n, const double signal[],
                 double arrivalTimeRelativeToStart);
    /// @result True indicates the velocity signal was computed.
    [[nodiscard]] bool haveVelocitySignal() const noexcept;
    /// @result The velocity signal with units of micrometers/second.
    /// @throws std::runtime_error if \c haveVelocitySignal() is false.
    [[nodiscard]] std::vector<double> getVelocitySignal() const; 
    /// @param[out] velocitySignal  The velocity signal with units of
    ///                             mirometers/second.
    /// @throws std::runtime_error if \c haveVelocitySignal() is false.
    void getVelocitySignal(std::vector<double> *velocitySignal) const;
    /// @result The absolute maximum peak ground velocity in micrometers/second.
    /// @throws std::runtime_error if \c haveVelocitySignal() is false. 
    [[nodiscard]] double getAbsoluteMaximumPeakGroundVelocity() const;

    /// @brief Reset class.
    void clear() noexcept;
    /// @brief Destructor.
    ~Preprocess();

    Preprocess(const Preprocess &) = delete;
    Preprocess(Preprocess &&) noexcept = delete;
    Preprocess& operator=(const Preprocess &) = delete;
    Preprocess& operator=(Preprocess &&) noexcept = delete;
private:
    class PreprocessImpl;
    std::unique_ptr<PreprocessImpl> pImpl;
};
}
#endif
