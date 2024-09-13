#pragma once

#include <string>
#include <vector>
#include <map>

class Logger
{
public:
    static Logger global;
    
    static Logger& Global()
    {
        return global;
    }

    void SetExpectedSize(unsigned int count);
    void PushValue(const std::string& name, double value);

    void ExportCSV(const std::string& filename) const;
private:  
    unsigned int expectedSize;
    std::map<std::string, std::vector<double>> values;
};