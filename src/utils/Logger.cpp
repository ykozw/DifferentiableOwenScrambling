#include "Logger.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

Logger Logger::global{};

void Logger::SetExpectedSize(unsigned int count)
{
    expectedSize = count;
    for (auto& it : values) it.second.reserve(count);
}

void Logger::PushValue(const std::string& name, double value)
{
    auto it = values.find(name);
    if (it == values.end())
    {
        it = values.insert(std::make_pair(name, std::vector<double>())).first;
        it->second.reserve(expectedSize);
    }

    it->second.push_back(value);
}

void Logger::ExportCSV(const std::string& filename) const
{
    if (values.size() == 0) return;

    std::ofstream out(filename);

    if (out.is_open())
    {
        out << std::fixed << std::setprecision(20);

        unsigned int max_size = 0;
        for (const auto& it : values) max_size = std::max(max_size, (unsigned int)it.second.size());

        // Header
        out << values.begin()->first;
        for (auto it = ++values.begin(); it != values.end(); it++)
            out << "," << it->first;
        out << '\n';

        for (unsigned int i = 0; i < max_size; i++)
        {
            auto it = values.begin();

            if (it->second.size() <= i) out << "nan";
            else                        out << it->second[i];

            it++;
            for (; it != values.end(); it++)
            {
                out << ",";
                if (it->second.size() <= i) out << "nan";
                else                        out << it->second[i];
            }
            out << '\n';
        }
    }
}