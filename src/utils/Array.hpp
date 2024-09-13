#pragma once

#include <iostream>
#include <vector>
#include <array>

template<unsigned char Dim>
using Index = std::array<unsigned int, Dim>;

template<unsigned char Dim>
using Shape = Index<Dim>;

template<unsigned char Dim>
inline Shape<Dim> ComputeStrides(const Shape<Dim>& dim) 
{
    Shape<Dim> strides;
    strides[Dim - 1] = 1;
    for (int i = (Dim - 2); i >= 0; i--)
        strides[i] = dim[i + 1] * strides[i + 1];
    return strides;
}

template<typename T, unsigned char Dim>
class Array
{
public:
    Array(const Shape<Dim>& shapes) : shape(shapes), allocatedShape(shapes), strides(ComputeStrides<Dim>(shapes)) 
    { 
        data.resize(size(), static_cast<T>(0));
    }

    Array(const Array<T, Dim>& other) : shape(other.shape), allocatedShape(other.allocatedShape), strides(other.strides), data(other.data)
    { }

    void zeros()
    {
        std::fill(data.begin(), data.end(), static_cast<T>(0));
    }

    T& operator[](const Index<Dim>& index)        { return data[indexOf(index)]; }
    T  operator[](const Index<Dim>& index)  const { return data[indexOf(index)]; }

    T& operator[](const unsigned int index)       { return data[index]; }
    T  operator[](const unsigned int index) const { return data[index]; }

    const double* pointAt(unsigned int i) const { return &data[indexOf(Index<Dim>{i})]; }
          double* pointAt(unsigned int i)       { return &data[indexOf(Index<Dim>{i})]; }

    double  at(const Index<Dim>& index) const { return operator[](index); }
    double& at(const Index<Dim>& index)       { return operator[](index); }

    unsigned int indexOf(const Index<Dim>& index) const {   
        unsigned int idx = 0;
        for (unsigned int i = 0; i < Dim; i++)
            idx += index[i] * strides[i];
        return idx;
    }

    unsigned int size() const { 
        unsigned int count = 1;
        for (unsigned int s : shape) count *= s;
        return count;
    }
    
    Shape<Dim> shape;
    Shape<Dim> allocatedShape;
    Shape<Dim> strides;
    
    std::vector<T> data;
private:
};

template<typename T, unsigned char Dim>
inline std::ostream& operator<<(std::ostream& out, const Array<T, Dim>& array) {
    const unsigned int size = array.size();
    
    // Flatten out array
    if constexpr (Dim == 1 || Dim > 3)
    {
        out << '[' << array[0];
        for (unsigned int i = 1; i < size; i++) out << ", " << array[i];
        out << ']';
        return out;
    }

    if constexpr (Dim == 2)
    {
        out << "[\n";
        for (unsigned int i = 0; i < array.shape[0]; i++)
        {
            out << "\t[" << array[{i, 0}];
            for (unsigned int j = 1; j < array.shape[1]; j++)
                out << ", " << array[{i, j}];
            out << "],\n";
        }
        out << "]";
        return out; 
    }

    if constexpr (Dim == 3)
    {
        out << "[\n";
        for (unsigned int k = 0; k < array.shape[0]; k++)
        {
            out << "\t[\n";
            for (unsigned int i = 0; i < array.shape[1]; i++)
            {
                out << "\t\t[" << array[{k, i, 0}];
                for (unsigned int j = 1; j < array.shape[2]; j++)
                    out << ", " << array[{k, i, j}];
                out << "],\n";
            }
            out << "\t],\n";    
        }
        out << "]";
        return out;
    }

    return out;
}

using Params           = Array<double, 1>;
using PointArray       = Array<double, 2>;
using Matrix           = Array<double, 2>;
using FuzzyBinaryArray = Array<double, 3>;
using BinaryArray      = Array<int, 3>; // Array of int instead of bool: 
                                        //  [container.requirements.dataraces]/2 of standard
                                        //      - std::vector<bool> is not thread safe (and this is the only exception)...


bool LoadPts (const std::string& filename, PointArray& array);
void WritePts(const std::string& filename, const PointArray& array);