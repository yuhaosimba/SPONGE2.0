#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "SPONGE/PM_force/esp_pswf.h"

namespace
{
float Eval_Table(const std::vector<float>& table, float x)
{
    if (x <= 0.0f) return table.front();
    if (x >= 1.0f) return table.back();
    float scaled = x * (table.size() - 1);
    int lower = static_cast<int>(scaled);
    int upper = lower + 1;
    if (upper >= static_cast<int>(table.size())) upper = table.size() - 1;
    float t = scaled - lower;
    return (1.0f - t) * table[lower] + t * table[upper];
}

float Eval_Poly(const std::vector<float>& coeff, float x)
{
    float value = 0.0f;
    for (int i = static_cast<int>(coeff.size()) - 1; i >= 0; i--)
    {
        value = value * x + coeff[i];
    }
    return value;
}

void Require(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "ESP PSWF smoke failed: " << message << "\n";
        std::exit(1);
    }
}

void Check_Close(float lhs, float rhs, float tolerance,
                 const std::string& message)
{
    float scale = std::max(1.0f, std::max(std::fabs(lhs), std::fabs(rhs)));
    if (std::fabs(lhs - rhs) > tolerance * scale)
    {
        std::cerr << "ESP PSWF smoke failed: " << message << " lhs=" << lhs
                  << " rhs=" << rhs << " tolerance=" << tolerance << "\n";
        std::exit(1);
    }
}

void Check_Direct_Mode(const ESP_PSWF_Table& table, bool use_poly)
{
    PM_Direct_Parameters direct;
    direct.backend = ParticleMeshBackend::ESP;
    direct.table_points = table.table_points;
    direct.split_poly_order = table.split_poly_order;
    direct.use_polynomial_tables = use_poly ? 1 : 0;
    direct.cutoff = table.cutoff;
    direct.split_real_table = table.split_real_table.data();
    direct.split_real_derivative_table =
        table.split_real_derivative_table.data();
    direct.split_real_coeff = table.split_real_coeff.data();
    direct.split_real_derivative_coeff =
        table.split_real_derivative_coeff.data();

    for (float r_factor : {1.0e-7f, 5.0e-4f, 0.05f, 0.2f, 0.5f, 0.8f, 0.99f})
    {
        float r = r_factor * table.cutoff;
        for (float charge_product : {-1.25f, 0.75f})
        {
            float split = ESP_Split_Long_Range_Factor(r, direct);
            float direct_energy =
                ESP_Get_Direct_Coulomb_Energy(charge_product, r, direct);
            float excluded_energy =
                ESP_Get_Excluded_Coulomb_Energy(charge_product, r, direct);
            float long_energy = charge_product * split / r;
            float full_energy = charge_product / r;
            Check_Close(direct_energy + long_energy, full_energy, 2.0e-4f,
                        "direct short plus PSWF long energy must be full "
                        "Coulomb");
            Check_Close(excluded_energy + long_energy, 0.0f, 2.0e-4f,
                        "excluded correction must cancel PSWF long energy");

            float direct_force =
                ESP_Get_Direct_Coulomb_Force(charge_product, r, direct);
            float excluded_force =
                ESP_Get_Excluded_Coulomb_Force(charge_product, r, direct);
            float full_force = charge_product / (r * r * r);
            Check_Close(direct_force + excluded_force, full_force, 2.0e-4f,
                        "direct short plus excluded-style long force must be "
                        "full Coulomb force coefficient");
        }
    }
}
}  // namespace

int main()
{
    ESP_Parameters parameters;
    parameters.tolerance = 1.0e-5f;
    parameters.cutoff = 8.0f;
    parameters.order = 8;
    parameters.table_points = 1024;

    ESP_PSWF_Table table = Build_ESP_PSWF_Table(parameters);

    Require(table.order == 8, "order should follow manual request");
    Require(table.table_points == 1024, "table point count should be stable");
    Require(std::isfinite(table.c_spread) && table.c_spread > 0.0f,
            "c_spread must be positive and finite");
    Require(std::isfinite(table.c_split) && table.c_split > 0.0f,
            "c_split must be positive and finite");
    Require(std::isfinite(table.self_energy_coeff) &&
                table.self_energy_coeff > 0.0f,
            "self energy coefficient must be positive and finite");

    Check_Close(table.split_real_table.front(), 0.0f, 1.0e-5f,
                "real-space long-range split must start at zero");
    Check_Close(table.split_real_table.back(), 1.0f, 1.0e-5f,
                "real-space long-range split must end at one");

    float max_real_poly_error = 0.0f;
    float max_real_derivative_poly_error = 0.0f;
    for (int i = 1; i < 64; i++)
    {
        float x = i / 64.0f;
        float real_table = Eval_Table(table.split_real_table, x);
        float real_poly = Eval_Poly(table.split_real_coeff, x);
        float derivative_table =
            Eval_Table(table.split_real_derivative_table, x);
        float derivative_poly = Eval_Poly(table.split_real_derivative_coeff, x);
        Require(std::isfinite(real_table) && std::isfinite(real_poly),
                "split values must be finite");
        Require(
            std::isfinite(derivative_table) && std::isfinite(derivative_poly),
            "split derivative values must be finite");
        Require(std::fabs(real_table) < 4.0f,
                "split real table should remain bounded");
        Require(std::fabs(derivative_table) < 50.0f,
                "split derivative table should remain bounded");
        max_real_poly_error =
            std::max(max_real_poly_error, std::fabs(real_table - real_poly));
        max_real_derivative_poly_error =
            std::max(max_real_derivative_poly_error,
                     std::fabs(derivative_table - derivative_poly));
    }
    Require(max_real_poly_error < 2.0e-3f,
            "split real polynomial approximation is too loose");
    Require(max_real_derivative_poly_error < 5.0e-2f,
            "split derivative polynomial approximation is too loose");

    Check_Direct_Mode(table, false);
    Check_Direct_Mode(table, true);

    std::cout << "ESP PSWF smoke passed: c_spread=" << table.c_spread
              << " c_split=" << table.c_split
              << " self=" << table.self_energy_coeff
              << " max_real_poly_error=" << max_real_poly_error
              << " max_real_derivative_poly_error="
              << max_real_derivative_poly_error << "\n";
    return 0;
}
