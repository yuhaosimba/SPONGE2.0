#include "esp_pswf.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace
{
constexpr double kPi = 3.141592653589793238462643383279502884;

double Clamp_Tolerance(double eps)
{
    if (eps < 1.0e-18) return 1.0e-18;
    if (eps > 1.0) return 1.0;
    return eps;
}

double Prolate_C_From_Table(double eps)
{
    static const double cs[180] = {
        0.43368E-16, 0.10048E+01, 0.17298E+01, 0.22271E+01, 0.26382E+01,
        0.30035E+01, 0.33409E+01, 0.36598E+01, 0.39658E+01, 0.42621E+01,
        0.45513E+01, 0.48347E+01, 0.51136E+01, 0.53887E+01, 0.56606E+01,
        0.59299E+01, 0.61968E+01, 0.64616E+01, 0.67247E+01, 0.69862E+01,
        0.72462E+01, 0.75049E+01, 0.77625E+01, 0.80189E+01, 0.82744E+01,
        0.85289E+01, 0.87826E+01, 0.90355E+01, 0.92877E+01, 0.95392E+01,
        0.97900E+01, 0.10040E+02, 0.10290E+02, 0.10539E+02, 0.10788E+02,
        0.11036E+02, 0.11284E+02, 0.11531E+02, 0.11778E+02, 0.12024E+02,
        0.12270E+02, 0.12516E+02, 0.12762E+02, 0.13007E+02, 0.13251E+02,
        0.13496E+02, 0.13740E+02, 0.13984E+02, 0.14228E+02, 0.14471E+02,
        0.14714E+02, 0.14957E+02, 0.15200E+02, 0.15443E+02, 0.15685E+02,
        0.15927E+02, 0.16169E+02, 0.16411E+02, 0.16652E+02, 0.16894E+02,
        0.17135E+02, 0.17376E+02, 0.17617E+02, 0.17858E+02, 0.18098E+02,
        0.18339E+02, 0.18579E+02, 0.18819E+02, 0.19059E+02, 0.19299E+02,
        0.19539E+02, 0.19778E+02, 0.20018E+02, 0.20257E+02, 0.20496E+02,
        0.20736E+02, 0.20975E+02, 0.21214E+02, 0.21452E+02, 0.21691E+02,
        0.21930E+02, 0.22168E+02, 0.22407E+02, 0.22645E+02, 0.22884E+02,
        0.23122E+02, 0.23360E+02, 0.23598E+02, 0.23836E+02, 0.24074E+02,
        0.24311E+02, 0.24549E+02, 0.24787E+02, 0.25024E+02, 0.25262E+02,
        0.25499E+02, 0.25737E+02, 0.25974E+02, 0.26211E+02, 0.26448E+02,
        0.26685E+02, 0.26922E+02, 0.27159E+02, 0.27396E+02, 0.27633E+02,
        0.27870E+02, 0.28106E+02, 0.28343E+02, 0.28580E+02, 0.28816E+02,
        0.29053E+02, 0.29289E+02, 0.29526E+02, 0.29762E+02, 0.29998E+02,
        0.30234E+02, 0.30471E+02, 0.30707E+02, 0.30943E+02, 0.31179E+02,
        0.31415E+02, 0.31651E+02, 0.31887E+02, 0.32123E+02, 0.32358E+02,
        0.32594E+02, 0.32830E+02, 0.33066E+02, 0.33301E+02, 0.33537E+02,
        0.33773E+02, 0.34008E+02, 0.34244E+02, 0.34479E+02, 0.34714E+02,
        0.34950E+02, 0.35185E+02, 0.35421E+02, 0.35656E+02, 0.35891E+02,
        0.36126E+02, 0.36362E+02, 0.36597E+02, 0.36832E+02, 0.37067E+02,
        0.37302E+02, 0.37537E+02, 0.37772E+02, 0.38007E+02, 0.38242E+02,
        0.38477E+02, 0.38712E+02, 0.38947E+02, 0.39181E+02, 0.39416E+02,
        0.39651E+02, 0.39886E+02, 0.40120E+02, 0.40355E+02, 0.40590E+02,
        0.40824E+02, 0.41059E+02, 0.41294E+02, 0.41528E+02, 0.41763E+02,
        0.41997E+02, 0.42232E+02, 0.42466E+02, 0.42700E+02, 0.42935E+02,
        0.43169E+02, 0.43404E+02, 0.43638E+02, 0.43872E+02, 0.44107E+02,
        0.44341E+02, 0.44575E+02, 0.44809E+02, 0.45044E+02, 0.45278E+02};

    eps = Clamp_Tolerance(eps);
    double d = -std::log10(eps);
    int i = static_cast<int>(d * 10.0 + 0.1);
    if (i < 1) i = 1;
    if (i > 180) i = 180;
    return cs[i - 1];
}

void Gauss_Legendre(int n, std::vector<double>& x, std::vector<double>& w)
{
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    const int m = (n + 1) / 2;
    for (int i = 0; i < m; i++)
    {
        double z = std::cos(kPi * (i + 0.75) / (n + 0.5));
        double z1 = 0.0;
        double pp = 0.0;
        do
        {
            double p1 = 1.0;
            double p2 = 0.0;
            for (int j = 1; j <= n; j++)
            {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (std::abs(z - z1) > 1.0e-14);

        x[i] = -z;
        x[n - 1 - i] = z;
        const double wi = 2.0 / ((1.0 - z * z) * pp * pp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
}

class Pswf0Reference
{
   public:
    Pswf0Reference(double c, int nquad) : c_(c)
    {
        Gauss_Legendre(nquad, nodes_, weights_);
        const int n = static_cast<int>(nodes_.size());
        std::vector<double> z(n, 0.0), next(n, 0.0);
        for (int i = 0; i < n; i++)
        {
            z[i] = std::sqrt(weights_[i]);
        }

        for (int iter = 0; iter < 250; iter++)
        {
            std::fill(next.begin(), next.end(), 0.0);
            for (int i = 0; i < n; i++)
            {
                const double swi = std::sqrt(weights_[i]);
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                {
                    sum += std::cos(c_ * nodes_[i] * nodes_[j]) *
                           std::sqrt(weights_[j]) * z[j];
                }
                next[i] = swi * sum;
            }
            double norm = 0.0;
            for (double v : next) norm += v * v;
            norm = std::sqrt(norm);
            if (norm <= std::numeric_limits<double>::min())
            {
                throw std::runtime_error("ESP PSWF power iteration failed");
            }
            for (double& v : next) v /= norm;

            double diff = 0.0;
            for (int i = 0; i < n; i++)
            {
                diff = std::max(diff, std::abs(next[i] - z[i]));
            }
            z.swap(next);
            if (diff < 1.0e-13) break;
        }

        std::vector<double> az(n, 0.0);
        for (int i = 0; i < n; i++)
        {
            const double swi = std::sqrt(weights_[i]);
            double sum = 0.0;
            for (int j = 0; j < n; j++)
            {
                sum += std::cos(c_ * nodes_[i] * nodes_[j]) *
                       std::sqrt(weights_[j]) * z[j];
            }
            az[i] = swi * sum;
        }
        eigenvalue_ = 0.0;
        for (int i = 0; i < n; i++) eigenvalue_ += z[i] * az[i];
        if (eigenvalue_ < 0.0)
        {
            eigenvalue_ = -eigenvalue_;
            for (double& v : z) v = -v;
        }
        if (std::abs(eigenvalue_) <= std::numeric_limits<double>::min())
        {
            throw std::runtime_error("ESP PSWF eigenvalue is zero");
        }

        values_.resize(n);
        for (int i = 0; i < n; i++) values_[i] = z[i] / std::sqrt(weights_[i]);
        const double psi0 = value_raw(0.0);
        if (std::abs(psi0) <= std::numeric_limits<double>::min())
        {
            throw std::runtime_error("ESP PSWF normalization failed");
        }
        for (double& v : values_) v /= psi0;
    }

    double value(double x) const
    {
        if (std::abs(x) > 1.0) return 0.0;
        return value_raw(x);
    }

    double derivative(double x) const
    {
        if (std::abs(x) > 1.0) return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < nodes_.size(); i++)
        {
            sum += weights_[i] * values_[i] * nodes_[i] *
                   std::sin(c_ * x * nodes_[i]);
        }
        return -c_ * sum / eigenvalue_;
    }

    double integral0(double r) const
    {
        if (r <= 0.0) return 0.0;
        if (r > 1.0) r = 1.0;
        double sum = 0.0;
        for (size_t i = 0; i < nodes_.size(); i++)
        {
            const double y = nodes_[i];
            double sinc = r;
            if (std::abs(c_ * y) > 1.0e-12)
            {
                sinc = std::sin(c_ * r * y) / (c_ * y);
            }
            sum += weights_[i] * values_[i] * sinc;
        }
        return sum / eigenvalue_;
    }

   private:
    double value_raw(double x) const
    {
        double sum = 0.0;
        for (size_t i = 0; i < nodes_.size(); i++)
        {
            sum += weights_[i] * values_[i] * std::cos(c_ * x * nodes_[i]);
        }
        return sum / eigenvalue_;
    }

    double c_;
    double eigenvalue_ = 0.0;
    std::vector<double> nodes_;
    std::vector<double> weights_;
    std::vector<double> values_;
};

double Compute_Lambda(const Pswf0Reference& pswf, double c)
{
    std::vector<double> xs, ws;
    Gauss_Legendre(160, xs, ws);
    double lambda = 0.0;
    for (size_t i = 0; i < xs.size(); i++)
    {
        lambda += ws[i] * pswf.value(xs[i]) * std::cos(c * xs[i] * 0.5);
    }
    const double denom = pswf.value(0.5);
    if (std::abs(denom) <= std::numeric_limits<double>::min())
    {
        throw std::runtime_error("ESP PSWF lambda normalization failed");
    }
    return lambda / denom;
}

int Default_Order_For_Tolerance(float tolerance)
{
    if (tolerance <= 1.0e-5f) return 8;
    if (tolerance <= 1.0e-4f) return 6;
    return 5;
}

float Linear_Table_Eval(const std::vector<float>& table, int offset, int count,
                        double x)
{
    if (x <= 0.0) return table[offset];
    if (x >= 1.0) return table[offset + count - 1];
    const double scaled = x * (count - 1);
    const int lo = static_cast<int>(scaled);
    const int hi = std::min(lo + 1, count - 1);
    const double t = scaled - lo;
    return static_cast<float>((1.0 - t) * table[offset + lo] +
                              t * table[offset + hi]);
}

std::vector<float> Interpolate_Monomial_Coefficients(
    int order, const std::function<double(double)>& func)
{
    std::vector<double> a(order * order, 0.0);
    std::vector<double> b(order, 0.0);
    for (int row = 0; row < order; row++)
    {
        const double x =
            0.5 *
            (1.0 - std::cos(kPi * (row + 0.5) / static_cast<double>(order)));
        double pow_x = 1.0;
        for (int col = 0; col < order; col++)
        {
            a[row * order + col] = pow_x;
            pow_x *= x;
        }
        b[row] = func(x);
    }

    for (int col = 0; col < order; col++)
    {
        int pivot = col;
        for (int row = col + 1; row < order; row++)
        {
            if (std::abs(a[row * order + col]) >
                std::abs(a[pivot * order + col]))
            {
                pivot = row;
            }
        }
        if (std::abs(a[pivot * order + col]) <=
            std::numeric_limits<double>::epsilon())
        {
            throw std::runtime_error("ESP monomial interpolation failed");
        }
        if (pivot != col)
        {
            for (int j = col; j < order; j++)
            {
                std::swap(a[col * order + j], a[pivot * order + j]);
            }
            std::swap(b[col], b[pivot]);
        }
        const double inv_pivot = 1.0 / a[col * order + col];
        for (int j = col; j < order; j++) a[col * order + j] *= inv_pivot;
        b[col] *= inv_pivot;

        for (int row = 0; row < order; row++)
        {
            if (row == col) continue;
            const double factor = a[row * order + col];
            if (factor == 0.0) continue;
            for (int j = col; j < order; j++)
            {
                a[row * order + j] -= factor * a[col * order + j];
            }
            b[row] -= factor * b[col];
        }
    }

    std::vector<float> coeff(order);
    for (int i = 0; i < order; i++) coeff[i] = static_cast<float>(b[i]);
    return coeff;
}

float Eval_Monomial(const std::vector<float>& coeff, int offset, int order,
                    double x)
{
    double y = 0.0;
    for (int i = order - 1; i >= 0; i--) y = y * x + coeff[offset + i];
    return static_cast<float>(y);
}

void Append_Coefficients(std::vector<float>& dst, const std::vector<float>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}
}  // namespace

float ESP_Get_Prolate_C(float tolerance)
{
    return static_cast<float>(Prolate_C_From_Table(tolerance));
}

ESP_PSWF_Table Build_ESP_PSWF_Table(const ESP_Parameters& parameters)
{
    ESP_PSWF_Table table;
    table.tolerance = parameters.tolerance;
    table.cutoff = parameters.cutoff;
    table.order = parameters.order > 0
                      ? parameters.order
                      : Default_Order_For_Tolerance(parameters.tolerance);
    table.table_points = std::max(parameters.table_points, 2);
    table.spread_poly_order = std::min(16, std::max(6, table.order + 4));
    table.split_poly_order = 16;

    const double spread_tol = Clamp_Tolerance(parameters.tolerance * 0.5);
    table.c_spread = static_cast<float>(Prolate_C_From_Table(spread_tol));
    table.c_split =
        static_cast<float>(Prolate_C_From_Table(parameters.tolerance));

    Pswf0Reference spread_pswf(table.c_spread, 128);
    Pswf0Reference split_pswf(table.c_split, 128);

    const double split_c0 = split_pswf.integral0(1.0);
    const double split_lambda = Compute_Lambda(split_pswf, table.c_split);
    table.c0_split = static_cast<float>(split_c0);
    table.psi0_split = static_cast<float>(split_pswf.value(0.0));
    table.lambda_split = static_cast<float>(split_lambda);
    table.lambda_spread =
        static_cast<float>(Compute_Lambda(spread_pswf, table.c_spread));
    table.self_energy_coeff = static_cast<float>(
        0.5 * table.psi0_split / (table.c0_split * parameters.cutoff));

    const int n = table.table_points;
    table.spread_window_table.resize(table.order * n);
    table.spread_window_derivative_table.resize(table.order * n);
    table.spread_window_fourier_table.resize(n);
    table.split_real_table.resize(n);
    table.split_real_derivative_table.resize(n);
    table.split_fourier_table.resize(n);
    table.split_fourier_derivative_table.resize(n);

    auto spread_value = [&](int i, double x)
    {
        const double arg = (x - table.order / 2.0 + i) / (table.order / 2.0);
        return spread_pswf.value(arg);
    };
    auto spread_derivative = [&](int i, double x)
    {
        const double arg = (x - table.order / 2.0 + i) / (table.order / 2.0);
        return spread_pswf.derivative(arg) * 2.0 / table.order;
    };
    auto spread_fourier = [&](double x)
    { return table.lambda_spread * spread_pswf.value(x); };
    auto split_real = [&](double x)
    { return split_pswf.integral0(x) / split_c0; };
    auto split_real_derivative = [&](double x)
    { return split_pswf.value(x) / split_c0; };
    auto split_fourier = [&](double x)
    { return split_lambda * split_pswf.value(x) / split_c0; };
    auto split_fourier_derivative = [&](double x)
    { return split_lambda * x * split_pswf.derivative(x) / split_c0; };

    for (int j = 0; j < n; j++)
    {
        const double x = static_cast<double>(j) / (n - 1);
        for (int i = 0; i < table.order; i++)
        {
            table.spread_window_table[i * n + j] =
                static_cast<float>(spread_value(i, x));
            table.spread_window_derivative_table[i * n + j] =
                static_cast<float>(spread_derivative(i, x));
        }
        table.spread_window_fourier_table[j] =
            static_cast<float>(spread_fourier(x));
        table.split_real_table[j] = static_cast<float>(split_real(x));
        table.split_real_derivative_table[j] =
            static_cast<float>(split_real_derivative(x));
        table.split_fourier_table[j] = static_cast<float>(split_fourier(x));
        table.split_fourier_derivative_table[j] =
            static_cast<float>(split_fourier_derivative(x));
    }

    for (int i = 0; i < table.order; i++)
    {
        Append_Coefficients(table.spread_window_coeff,
                            Interpolate_Monomial_Coefficients(
                                table.spread_poly_order,
                                [&](double x) { return spread_value(i, x); }));
        Append_Coefficients(table.spread_window_derivative_coeff,
                            Interpolate_Monomial_Coefficients(
                                table.spread_poly_order, [&](double x)
                                { return spread_derivative(i, x); }));
    }
    table.spread_window_fourier_coeff = Interpolate_Monomial_Coefficients(
        table.split_poly_order, [&](double x) { return spread_fourier(x); });
    table.split_real_coeff = Interpolate_Monomial_Coefficients(
        table.split_poly_order, [&](double x) { return split_real(x); });
    table.split_real_derivative_coeff =
        Interpolate_Monomial_Coefficients(table.split_poly_order, [&](double x)
                                          { return split_real_derivative(x); });
    table.split_fourier_coeff = Interpolate_Monomial_Coefficients(
        table.split_poly_order, [&](double x) { return split_fourier(x); });
    table.split_fourier_derivative_coeff = Interpolate_Monomial_Coefficients(
        table.split_poly_order,
        [&](double x) { return split_fourier_derivative(x); });

    const int checks = std::max(256, table.table_points * 2);
    for (int j = 0; j <= checks; j++)
    {
        const double x = (j + 0.5) / (checks + 1.0);
        for (int i = 0; i < table.order; i++)
        {
            const float tab =
                Linear_Table_Eval(table.spread_window_table, i * n, n, x);
            const float poly = Eval_Monomial(table.spread_window_coeff,
                                             i * table.spread_poly_order,
                                             table.spread_poly_order, x);
            const float ref = static_cast<float>(spread_value(i, x));
            table.max_window_table_error =
                std::max(table.max_window_table_error, std::abs(tab - ref));
            table.max_window_poly_error =
                std::max(table.max_window_poly_error, std::abs(poly - ref));
        }
        const float split_tab =
            Linear_Table_Eval(table.split_fourier_table, 0, n, x);
        const float split_poly = Eval_Monomial(table.split_fourier_coeff, 0,
                                               table.split_poly_order, x);
        const float split_ref = static_cast<float>(split_fourier(x));
        table.max_split_table_error = std::max(table.max_split_table_error,
                                               std::abs(split_tab - split_ref));
        table.max_split_poly_error = std::max(table.max_split_poly_error,
                                              std::abs(split_poly - split_ref));
    }

    return table;
}
