#ifndef PERFORMANCE_TIMER_H
#define PERFORMANCE_TIMER_H

#include <chrono>

class PerformanceTimer {
public:
    void start();
    void stop();
    double getElapsedSeconds() const;

private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_end;
};

#endif // PERFORMANCE_TIMER_H
