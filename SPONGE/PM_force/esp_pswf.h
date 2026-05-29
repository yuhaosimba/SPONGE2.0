#pragma once

#include <vector>

#include "PM_force.h"

struct ESP_PSWF_Table
{
    int order = 0;
    int table_points = 0;
    int spread_poly_order = 0;
    int split_poly_order = 0;
    float tolerance = 0.0f;
    float cutoff = 0.0f;
    float c_spread = 0.0f;
    float c_split = 0.0f;
    float c0_split = 0.0f;
    float psi0_split = 0.0f;
    float lambda_split = 0.0f;
    float lambda_spread = 0.0f;
    float self_energy_coeff = 0.0f;
    float max_window_table_error = 0.0f;
    float max_split_table_error = 0.0f;
    float max_window_poly_error = 0.0f;
    float max_split_poly_error = 0.0f;

    std::vector<float> spread_window_table;
    std::vector<float> spread_window_derivative_table;
    std::vector<float> spread_window_coeff;
    std::vector<float> spread_window_derivative_coeff;
    std::vector<float> spread_window_fourier_table;
    std::vector<float> spread_window_fourier_coeff;

    std::vector<float> split_real_table;
    std::vector<float> split_real_derivative_table;
    std::vector<float> split_real_coeff;
    std::vector<float> split_real_derivative_coeff;
    std::vector<float> split_fourier_table;
    std::vector<float> split_fourier_derivative_table;
    std::vector<float> split_fourier_coeff;
    std::vector<float> split_fourier_derivative_coeff;
};

float ESP_Get_Prolate_C(float tolerance);
ESP_PSWF_Table Build_ESP_PSWF_Table(const ESP_Parameters& parameters);
