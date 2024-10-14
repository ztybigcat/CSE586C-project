#include "performance_timer.h"

void PerformanceTimer::start() {
    m_start = std::chrono::high_resolution_clock::now();
}

void PerformanceTimer::stop() {
    m_end = std::chrono::high_resolution_clock::now();
}

double PerformanceTimer::getElapsedSeconds() const {
    std::chrono::duration<double> elapsed = m_end - m_start;
    return elapsed.count();
}
