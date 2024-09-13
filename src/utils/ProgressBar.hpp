#pragma once

#include <chrono>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

class ProgressBar
{
public:
    class PBarIterator
    {
    public:
        PBarIterator(const ProgressBar* pbar, unsigned int index) : 
            ptr(pbar), idx(index)  
        {}

        PBarIterator operator++() { ++idx; ptr->Draw(idx); return *this; }
        bool operator!=(const PBarIterator& other) { return other.idx != idx; }
        unsigned int operator*() const { return idx; }
    private:
        const ProgressBar* ptr;
        unsigned int idx = 0;
    };
    
    ProgressBar(unsigned int end) : _start(0), _end(end)
    { } 

    ProgressBar(unsigned int start, unsigned int end) : _start(start), _end(end)
    { }

    void Draw(unsigned int index) const
    {
        // Time computation
        auto current = std::chrono::high_resolution_clock::now();

        const auto it_time = 
            std::chrono::duration_cast<std::chrono::microseconds>(current - last_it).count();
        
        const auto current_time = 
            std::chrono::duration_cast<std::chrono::microseconds>(current -   start).count();
        const auto current_time_second = (current_time / 1000000) % 60;
        const auto current_time_minuts =  current_time / 60000000;

        const auto its_second = it_time / 1000000.0;
        const auto total_time_pred   = current_time + (_end - index) * it_time;
        const auto time_pred_seconds = (total_time_pred / 1000000) % 60;
        const auto time_pred_minuts  =  total_time_pred / 60000000;


        const double percentage = (double)(index) / (double)(_end - _start);
        unsigned int val  = static_cast<unsigned int>(percentage * 100);
        unsigned int lpad = static_cast<unsigned int>(percentage * PBWIDTH);
        unsigned int rpad = static_cast<unsigned int>(PBWIDTH - lpad);

        std::printf("\r%3d%% [%.*s%*s] [%02ld:%02ld<%02ld:%02ld] [%.2fs/it]", val, lpad, PBSTR, rpad, "", current_time_minuts, current_time_second, time_pred_minuts, time_pred_seconds, its_second);
        
        std::fflush(stdout);
        last_it = current;

        if (index == _end) std::printf("\n");
    }

    PBarIterator begin() const 
    {
        start   = std::chrono::high_resolution_clock::now(); 
        last_it = std::chrono::high_resolution_clock::now();
        return PBarIterator(this, _start); 
    }
    PBarIterator end()   const { return PBarIterator(this, _end); }
private:
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> start;
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> last_it;

    const unsigned int _start;
    const unsigned int _end;
};