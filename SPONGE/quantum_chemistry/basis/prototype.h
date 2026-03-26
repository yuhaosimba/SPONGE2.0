#pragma once

#include <map>
#include <string>
#include <vector>

struct QC_SHELL_DATA
{
    int l;
    std::vector<float> exps;
    std::vector<float> coeffs;
};

// Use this type alias for the basis data map
using QC_BASIS_MAP = std::map<std::string, std::vector<QC_SHELL_DATA>>;

struct QC_BASIS_SET
{
    virtual ~QC_BASIS_SET() = default;
    virtual void Initialize() = 0;

    const char* name = "";   // e.g. "6-31g", "def2-qzvp"
    bool spherical = false;  // whether basis uses spherical harmonics
    QC_BASIS_MAP data;
};
