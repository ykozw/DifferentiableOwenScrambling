#include "Array.hpp"

#include <fstream>
#include <iomanip>

bool LoadPts(const std::string& filename, PointArray& array)
{
    std::ifstream input(filename.c_str());
    if (input.is_open())
    {
        for (unsigned int i = 0; i < array.shape[0]; i++)
        {
            for (unsigned int d = 0; d < array.shape[1]; d++)
            {
                input >> array[{i, d}];
            }
        }

        return true;
    }
    else
    {
        std::cout << "File not found !" << std::endl;
    }
    
    return false;
}

void WritePts(const std::string& filename, const PointArray& array)
{
    std::ofstream out(filename.c_str());
    out << std::setprecision(20) << std::fixed;

    for (unsigned int i = 0; i < array.shape[0]; i++)
    {
        for (unsigned int d = 0; d < array.shape[1]; d++)
        {
            out << array[{i, d}] << " ";
        }
        out << '\n';
    }
}