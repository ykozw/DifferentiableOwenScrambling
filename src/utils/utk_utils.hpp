

#pragma once

#include <cstring> // memcpy
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

// Pointset class

// Just a wrapper around a T*
// It can be set to be a view or hold its own memory
// The copy/view mecanism is introduced for seemless interaction with numpy (and python protocol buffer)
// This also allows to use the library with 'plain double pointers'.
template<typename T>
class Pointset
{
public:
    Pointset()
    { }

    Pointset(uint32_t n, uint32_t d) 
    {
        Resize(n, d);
    }

    static Pointset<T> Copy(const T* data, uint32_t n, uint32_t d)
    {
        Pointset<T> pt(n, d);
        std::memcpy(pt.data, data, sizeof(T) * n * d);
        return pt;
    }

    // Take a view ! (allow for simple passing and returns)
    Pointset(const Pointset& other)
    {
        N = other.N;
        D = other.D;
        C = other.C;
        isView = true;
        
        // Shared ptr copy, will inherit non deleter 
        // if needed
        data = other.data; 
    }

    // Take a view ! (allow for simple passing and returns)
    Pointset& operator=(const Pointset& other)
    {
        if (this != &other)
        {
            N = other.N;
            D = other.D;
            C = other.C;
            isView = true;

            data = other.data; 
        }
        return *this;
    }

    Pointset(Pointset&& other)
    {
        N = std::move(other.N);
        D = std::move(other.D);
        C = std::move(other.C);
        isView = std::move(other.isView);
        data = std::move(other.data); 
    }

    Pointset& operator=(Pointset&& other)
    {
        N = std::move(other.N);
        D = std::move(other.D);
        C = std::move(other.C);
        isView = std::move(other.isView);
        data = std::move(other.data); 
        return *this;
    }

    T* Data()
    {
        return data.data();
    }

    const T* Data() const
    {
        return data.data();
    }

    T* operator[](uint32_t i) 
    {
        return Data() + i * D;
    }

    const T* operator[](uint32_t i) const
    {
        return Data() + i * D;
    }

    void swap(Pointset<T>& other)
    {
        std::swap(other.N, N);
        std::swap(other.D, D);
        std::swap(other.isView, isView);
        std::swap(other.data, data);
    }

    uint32_t Npts() const
    {
        return N;
    }

    uint32_t Ndim() const
    {
        return D;
    }

    uint32_t Capacity() const
    {
        return C;
    }

    void Fill(T value = 0) const
    {
        std::fill(data.begin(), data.end(), value);
    }

    void Resize(uint32_t n, uint32_t d, uint32_t hint_Capacity = 0)
    {
        data.resize(n * d);
        N = n;
        D = d;
    }

    T& PushBack()
    {
        uint32_t n = N;
        Resize(N + 1, D, C * 2);

        return (*this)[n][0];
    }  

    ~Pointset()
    {  }
private:
    bool isView = false;

    uint32_t N = 0;
    uint32_t D = 1;
    uint32_t C = 0;

    std::vector<double> data; 
    // Can not set data's pointer... Or maybe with custom allocators
    // Anyway keeping the pointer is simpler.
    // std::vector<T> data; 
};

#pragma once

#include <iomanip>
#include <fstream>
#include <string>
#include <limits>

std::string rstrip(const std::string& s)
{
    std::string rslt = s;
    rslt.erase(rslt.find_last_not_of (" \n\r\t") + 1);
    return rslt;
}

template<class Stream, typename T>
inline void write_text_pointset_stream(Stream& st, const Pointset<T>& pts)
{
    for (uint32_t i = 0; i < pts.Npts(); i++)
    {
        for (uint32_t d = 0; d < pts.Ndim() - 1; d++)
        {
            st << pts[i][d] << ' ';
        }
        st << pts[i][pts.Ndim() - 1] << '\n';
    }
}
    
template<typename T>
inline bool write_text_pointset(const std::string& dest, const Pointset<T>& pts)
{
    std::ofstream file(dest); 
    
    if (!file.is_open()) return false;

    file << std::setprecision(std::numeric_limits<T>::digits10 + 2) << std::fixed;
    write_text_pointset_stream(file, pts);
    return true;
}

template<typename T>
inline bool write_text_pointsets(const std::string& dest, const std::vector<Pointset<T>>& pts)
{
    std::ofstream file(dest);
    if (!file.is_open()) return false;
    
    file << std::setprecision(std::numeric_limits<T>::digits10 + 2) << std::fixed;
    for (uint32_t i = 0; i < pts.size() - 1; i++)
    {
        write_text_pointset_stream(file, pts[i]);
        file << "#\n";
    }
    
    if (pts.size() != 0)
        write_text_pointset_stream(file, pts.back());

    return true;
}

template<class Stream, typename T>
inline Pointset<T> read_text_pointset_stream(Stream& st)
{
    // At least one element
    Pointset<T> pts = Pointset<T>(0, 1);
    std::string line = "#";

    // Skips comments (at the beginning only) (if any)
    while (line[0] == '#' && std::getline(st, line));

    // Check for end of file. Not sure if empty line is possible
    // Better safe than sorry
    if (st.eof() || line.empty())
        return pts;

    std::istringstream sstream(rstrip(line));
    while(sstream.good()) sstream >> pts.PushBack();
    uint32_t d = pts.Npts();

    while(std::getline(st, line))
    {
        if (line[0] == '#') break;
        std::istringstream tmp(rstrip(line));

        while(tmp.good()) tmp >> pts.PushBack();
        
    }

    pts.Resize(pts.Npts() / d, d);
    // pts.Shrink();
    // return pts;
    return pts;
}

template<class Stream, typename T>
inline std::vector<Pointset<T>> read_text_pointsets_stream(Stream& st)
{
    std::vector<Pointset<T>> pointsets;
    while (st.good())
    {
        Pointset<T> pointset = read_text_pointset_stream<decltype(st), T>(st);
        if (pointset.Npts() > 0)
            pointsets.push_back(pointset);
    }
    return pointsets;
}

template<typename T>
inline std::vector<Pointset<T>> read_text_pointset(const char* filepath)
{
    std::ifstream file(filepath);
    return read_text_pointsets_stream<decltype(file), T>(file);
}

template<typename T>
inline std::vector<Pointset<T>> read_pointsets(const char* filename)
{
    uint32_t length = strlen(filename);
    return read_text_pointset<T>(filename);
}