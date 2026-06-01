#include "PM_force.h"

#include <algorithm>
#include <cmath>
#include <exception>

#include "esp_pswf.h"

/*
    SPONGE particle-mesh implementation.
    Public entry points stay in Particle_Mesh, while reciprocal-space work
    dispatches between the traditional PME backend and the ESP/PSWF backend.
    The MPI FFT interfaces remain declared, but the current ESP path targets
    single-process execution.
*/

// constants
#define PI 3.1415926f
#define INVSQRTPI 0.56418958835977f
#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446f
static __device__ float PME_Ma[4] = {1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0};
static __device__ float PME_Mb[4] = {0, 0.5, -1, 0.5};
static __device__ float PME_Mc[4] = {0, 0.5, 0, -0.5};
static __device__ float PME_Md[4] = {0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
static __device__ float PME_dMa[4] = {0.5, -1.5, 1.5, -0.5};
static __device__ float PME_dMb[4] = {0, 1, -2, 1};
static __device__ float PME_dMc[4] = {0, 0.5, 0, -0.5};
static constexpr int ESP_ORDER5 = 5;
static constexpr int ESP_ORDER5_SUPPORT = 125;
static constexpr int ESP_GPU_SPREAD_LANES_PER_ATOM = 32;
static constexpr int ESP_GPU_SPREAD_ATOMS_PER_BLOCK = 4;
static constexpr int ESP_GPU_FINAL_LANES_PER_ATOM = 8;
static constexpr int ESP_GPU_FINAL_ATOMS_PER_BLOCK = 128;

struct ESP_Default_Selection
{
    float c_split = 0.0f;
    float c_window = 0.0f;
    float alpha = 0.0f;
    float target_grid_spacing = 0.0f;
    float effective_grid_spacing = 0.0f;
    int fftx = 0;
    int ffty = 0;
    int fftz = 0;
    int order = 0;
};

static int ESP_Default_Order_Gromacs_Style(float tolerance)
{
    if (tolerance <= 1.0e-5f) return 6;
    if (tolerance <= 1.0e-4f) return 5;
    return 4;
}

static float ESP_Default_Box_Length_Reference(VECTOR box_length)
{
    float ref = box_length.x;
    if (box_length.y > 0.0f) ref = std::min(ref, box_length.y);
    if (box_length.z > 0.0f) ref = std::min(ref, box_length.z);
    return ref;
}

static int ESP_Get_Friendly_Fft_Count(float target_count)
{
    int count = std::max(4, (int)ceilf(target_count));
    if (count % 2 != 0) count += 1;
    while (!Check_2357_Factor(count))
    {
        count += 2;
    }
    return count;
}

static float ESP_Solve_Window_Bandlimit(float c_split, float cutoff,
                                        float box_length_ref)
{
    const double ratio = 4.0 * sqrt(5.0) / pow((double)CONSTANT_Pi, 1.5);
    const double cs = std::max((double)c_split, 1.0);
    const double rc = std::max((double)cutoff, 1.0e-6);
    const double lref = std::max((double)box_length_ref, rc + 1.0e-6);
    const double rhs =
        cs + 0.5 * std::log(cs) - std::log(ratio) + 0.5 * std::log(lref / rc);

    double cw = std::max(rhs, 1.0);
    for (int iter = 0; iter < 32; iter++)
    {
        const double f = cw - 0.5 * std::log(cw) - rhs;
        const double df = 1.0 - 0.5 / cw;
        const double step = f / df;
        cw -= step;
        if (cw <= 0.5) cw = 0.5 + 1.0e-6;
        if (std::abs(step) <= 1.0e-12 * std::max(1.0, cw))
        {
            break;
        }
    }
    return (float)cw;
}

static ESP_Default_Selection Build_ESP_Default_Selection(float tolerance,
                                                         float cutoff,
                                                         VECTOR box_length)
{
    ESP_Default_Selection selection;
    const float clamped_tolerance =
        std::min(std::max(tolerance, 1.0e-12f), 0.5f);
    const float box_length_ref = ESP_Default_Box_Length_Reference(box_length);
    if (cutoff <= 0.0f || box_length_ref <= 0.0f)
    {
        return selection;
    }

    selection.c_split = std::max(1.0f, ESP_Get_Prolate_C(clamped_tolerance));
    selection.c_window =
        ESP_Solve_Window_Bandlimit(selection.c_split, cutoff, box_length_ref);
    selection.alpha = cutoff * selection.c_window / selection.c_split;
    selection.target_grid_spacing = CONSTANT_Pi * cutoff / selection.c_split;

    selection.fftx =
        ESP_Get_Friendly_Fft_Count(box_length.x / selection.target_grid_spacing);
    selection.ffty =
        ESP_Get_Friendly_Fft_Count(box_length.y / selection.target_grid_spacing);
    selection.fftz =
        ESP_Get_Friendly_Fft_Count(box_length.z / selection.target_grid_spacing);
    selection.effective_grid_spacing =
        std::min(box_length.x / selection.fftx,
                 std::min(box_length.y / selection.ffty,
                          box_length.z / selection.fftz));
    selection.order = ESP_Default_Order_Gromacs_Style(clamped_tolerance);
    return selection;
}

// 计算B样条插值的递归函数 (compact-window function, or SI kernel)
static float M_(float u, int n)
{
    if (n == 2)
    {
        if (u > 2 || u < 0) return 0;
        return 1 - abs(u - 1);
    }
    else
        return u / (n - 1) * M_(u, n - 1) +
               (n - u) / (n - 1) * M_(u - 1, n - 1);
}

// 修正B样条插值的递归函数 (influence function)
static float getb(int k, int NFFT, int B_order)
{
    FFT_COMPLEX tempc, tempc2, res;
    float tempf;
    REAL(tempc2) = 0;
    IMAGINARY(tempc2) = 0;

    REAL(tempc) = 0;
    IMAGINARY(tempc) = 2 * (B_order - 1) * PI * k / NFFT;
    res = expc(tempc);

    for (int kk = 0; kk < (B_order - 1); kk++)
    {
        REAL(tempc) = 0;
        IMAGINARY(tempc) = 2.0f * PI * k / NFFT * kk;
        tempc = expc(tempc);
        tempf = M_(kk + 1, B_order);
        REAL(tempc2) += tempf * REAL(tempc);
        IMAGINARY(tempc2) += tempf * IMAGINARY(tempc);
    }
    res = divc(res, tempc2);
    return REAL(res) * REAL(res) + IMAGINARY(res) * IMAGINARY(res);
}

// PMC_IZ Method
static __global__ void Build_PMC_IZ_C(const int PME_Nfft, int fftx, int ffty,
                                      int fftz,
                                      float box_length_inverse_x_square,
                                      float box_length_inverse_y_square,
                                      float grid_length_of_z, float beta,
                                      float scalor, FFT_COMPLEX* C)
{
    SIMPLE_DEVICE_FOR(tid, PME_Nfft)
    {
        int ffta = (fftx / 2 + 1);
        int grid_x = tid % ffta;
        int grid_y = (tid % (ffta * ffty)) / ffta;
        int grid_z = tid / ffty / ffta;
        if (grid_x >= fftx / 2)
        {
            grid_x = fftx - grid_x;
        }
        if (grid_y >= ffty / 2)
        {
            grid_y = ffty - grid_y;
        }
        if (grid_z >= fftz / 2)
        {
            grid_z = fftz - grid_z;
        }
        float z = grid_length_of_z * grid_z;
        float A = 2.0f * CONSTANT_Pi *
                  sqrtf(grid_x * grid_x * box_length_inverse_x_square +
                        grid_y * grid_y * box_length_inverse_y_square);
        float AB = A / beta / 2.0f;
        float zb2 = z * beta;
        float AB_minus_zb2 = AB - zb2;
        float AB_plus_zb2 = AB + zb2;
        float temp_f =
            expf(-A * z) * (erfcf(AB_minus_zb2) +
                            expf(2.0f * A * z - AB_plus_zb2 * AB_plus_zb2) *
                                erfcxf(AB_plus_zb2));
        temp_f = temp_f / A;
        if (grid_x == 0 && grid_y == 0)
        {
            temp_f =
                2.0f / sqrtf(CONSTANT_Pi) / beta * (1.0f - expf(-zb2 * zb2)) -
                2.0f * z * erff(zb2);
        }
        REAL(C[tid]) = scalor * temp_f;
        IMAGINARY(C[tid]) = 0.;
    }
}

static __global__ void Build_PMC_IZ_BC_Final(const int Nfft, int fftx, int ffty,
                                             int fftz, const FFT_COMPLEX* C,
                                             const FFT_COMPLEX* B, float* BC)
{
    SIMPLE_DEVICE_FOR(tid, Nfft)
    {
        int fftc = fftz / 2 + 1;
        int ffta = fftx / 2 + 1;
        int zi = tid % fftc;
        int yi = (tid / fftc) % ffty;
        int xi = tid / fftc / ffty;
        if (xi >= fftx / 2)
        {
            xi = fftx - xi;
        }
        int ti = zi * ffta * ffty + yi * ffta + xi;
        float b = REAL(B[ti]);
        BC[tid] = REAL(C[ti]) / (b * b);
    }
}

static void Build_PMC_IZ_BC(CONTROLLER* controller, int fftx, int ffty,
                            int fftz, int PME_Nfft, int PME_Nall, int PME_Nin,
                            float box_length_inverse_x_square,
                            float box_length_inverse_y_square,
                            float grid_length_of_z, float beta, float scalor,
                            float** BC)
{
    Device_Malloc_Safely((void**)BC, sizeof(float) * PME_Nfft);
    FFT_SIZE_t n2d[2] = {ffty, fftx};
    FFT_RESULT result;
    FFT_HANDLE plan_2d_many_c2r, plan_3d_temp_r2c;

    result = SPONGE_FFT_WRAPPER::Make_FFT_Plan(&plan_2d_many_c2r, fftz, 2, n2d,
                                               FFT_C2R);
    if (result != FFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed, "Build_PMC_IZ_Efficient_Potential",
            "Reason:\n\tFail to create the batched 2D FFT plan");
    }
    FFT_SIZE_t n3d[3] = {fftz, ffty, fftx};

    result = SPONGE_FFT_WRAPPER::Make_FFT_Plan(&plan_3d_temp_r2c, 1, 3, n3d,
                                               FFT_R2C);
    if (result != FFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(spongeErrorMallocFailed,
                                       "Build_PMC_IZ_Efficient_Potential",
                                       "Reason:\n\tFail to create the "
                                       "temporary 3D Real to Complex FFT plan");
    }
    FFT_COMPLEX *B, *C;
    float *d_FB, *h_FB, *FC;
    int temp_Nfft = (fftx / 2 + 1) * ffty * fftz;
    Device_Malloc_Safely((void**)&B, sizeof(FFT_COMPLEX) * temp_Nfft);
    Device_Malloc_Safely((void**)&C, sizeof(FFT_COMPLEX) * temp_Nfft);
    Device_Malloc_Safely((void**)&FC, sizeof(float) * PME_Nall);
    Malloc_Safely((void**)&h_FB, sizeof(float) * PME_Nall);

    for (int i = 0; i < PME_Nall; i = i + 1)
    {
        h_FB[i] = 0.;
    }
    float temp_b_spline[3] = {1. / 6., 2. / 3., 1. / 6.};
    for (int k = -1; k <= 1; k = k + 1)
    {
        for (int j = -1; j <= 1; j = j + 1)
        {
            for (int i = -1; i <= 1; i = i + 1)
            {
                float weight = temp_b_spline[k + 1] * temp_b_spline[j + 1] *
                               temp_b_spline[i + 1];
                int kk, jj, ii;
                if (k < 0)
                {
                    kk = k + fftz;
                }
                else
                {
                    kk = k;
                }
                if (j < 0)
                {
                    jj = j + ffty;
                }
                else
                {
                    jj = j;
                }
                if (i < 0)
                {
                    ii = i + fftx;
                }
                else
                {
                    ii = i;
                }
                h_FB[ii + jj * fftx + kk * fftx * ffty] = weight;
            }
        }
    }
    Device_Malloc_And_Copy_Safely((void**)&d_FB, h_FB,
                                  sizeof(float) * PME_Nall);
    SPONGE_FFT_WRAPPER::R2C(plan_3d_temp_r2c, d_FB, B);
    Launch_Device_Kernel(Build_PMC_IZ_C,
                         (temp_Nfft + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, temp_Nfft,
                         fftx, ffty, fftz, box_length_inverse_x_square,
                         box_length_inverse_y_square, grid_length_of_z, beta,
                         scalor, C);

    SPONGE_FFT_WRAPPER::C2R(plan_2d_many_c2r, C, FC);
    SPONGE_FFT_WRAPPER::R2C(plan_3d_temp_r2c, FC, C);
    Launch_Device_Kernel(Build_PMC_IZ_BC_Final,
                         (PME_Nfft + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, PME_Nfft, fftx,
                         ffty, fftz, C, B, BC[0]);

    Free_Single_Device_Pointer((void**)&FC);
    Free_Single_Device_Pointer((void**)&C);
    Free_Single_Device_Pointer((void**)&B);
    Free_Host_And_Device_Pointer((void**)&h_FB, (void**)&d_FB);
    SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&plan_2d_many_c2r);
    SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&plan_3d_temp_r2c);
}

// 根据截断距离和精度要求计算Ewald屏蔽参数beta
static float Get_Beta(float cutoff, float tolerance)
{
    float beta, low, high, tempf;
    int ilow, ihigh;

    high = 1.0;
    ihigh = 1;

    while (1)
    {
        tempf = erfc(high * cutoff) / cutoff;
        if (tempf <= tolerance) break;
        high *= 2;
        ihigh++;
    }

    ihigh += 50;
    low = 0.0;
    for (ilow = 1; ilow < ihigh; ilow++)
    {
        beta = (low + high) / 2;
        tempf = erfc(beta * cutoff) / cutoff;
        if (tempf >= tolerance)
            low = beta;
        else
            high = beta;
    }
    return beta;
}

// ene += factor * charge_sum^2
static __global__ void device_add(float* ene, float factor, float* charge_sum)
{
    ene[0] += factor * charge_sum[0] * charge_sum[0];
}

static __global__ void charge_square_kernel(int element_number,
                                            const float* charge,
                                            float* charge_square)
{
    SIMPLE_DEVICE_FOR(i, element_number)
    {
        float q = charge[i];
        charge_square[i] = q * q;
    }
}

static const char* Particle_Mesh_Backend_Name(ParticleMeshBackend backend)
{
    switch (backend)
    {
        case ParticleMeshBackend::ESP:
            return "esp";
        case ParticleMeshBackend::PME:
        default:
            return "pme";
    }
}

static const char* ESP_Parameter_Mode_Name(ESPParameterMode mode)
{
    switch (mode)
    {
        case ESPParameterMode::MANUAL:
            return "manual";
        case ESPParameterMode::AUTO:
        default:
            return "auto";
    }
}

static const char* ESP_Table_Mode_Name(ESPTableMode mode)
{
    switch (mode)
    {
        case ESPTableMode::TABLE:
            return "table";
        case ESPTableMode::POLY:
        default:
            return "poly";
    }
}

static void Parse_ESP_Parameters(CONTROLLER* controller,
                                 const char* module_name, ESP_Parameters* esp,
                                 float tolerance, float cutoff)
{
    esp->tolerance = tolerance;
    esp->cutoff = cutoff;

    if (controller->Command_Exist(module_name, "esp_tolerance"))
    {
        controller->Check_Float(module_name, "esp_tolerance",
                                "Particle_Mesh::Initial");
        esp->tolerance =
            atof(controller->Command(module_name, "esp_tolerance"));
        if (esp->tolerance <= 0.0f)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tesp_tolerance should be positive.");
        }
    }
    if (controller->Command_Exist(module_name, "esp_order"))
    {
        controller->Check_Int(module_name, "esp_order",
                              "Particle_Mesh::Initial");
        esp->order = atoi(controller->Command(module_name, "esp_order"));
        if (esp->order <= 0)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tesp_order should be positive.");
        }
    }
    if (controller->Command_Exist(module_name, "esp_grid_spacing"))
    {
        controller->Check_Float(module_name, "esp_grid_spacing",
                                "Particle_Mesh::Initial");
        esp->grid_spacing =
            atof(controller->Command(module_name, "esp_grid_spacing"));
        if (esp->grid_spacing <= 0.0f)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tesp_grid_spacing should be positive.");
        }
    }
    if (controller->Command_Exist(module_name, "esp_table_points"))
    {
        controller->Check_Int(module_name, "esp_table_points",
                              "Particle_Mesh::Initial");
        esp->table_points =
            atoi(controller->Command(module_name, "esp_table_points"));
        if (esp->table_points < 2)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tesp_table_points should be at least 2.");
        }
    }
    if (controller->Command_Exist(module_name, "esp_parameter_mode"))
    {
        const char* mode =
            controller->Command(module_name, "esp_parameter_mode");
        if (is_str_equal(mode, "auto"))
        {
            esp->parameter_mode = ESPParameterMode::AUTO;
        }
        else if (is_str_equal(mode, "manual"))
        {
            esp->parameter_mode = ESPParameterMode::MANUAL;
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tesp_parameter_mode should be 'auto' or 'manual'.");
        }
    }
    if (controller->Command_Exist(module_name, "esp_table_mode"))
    {
        const char* mode = controller->Command(module_name, "esp_table_mode");
        if (is_str_equal(mode, "poly"))
        {
            esp->table_mode = ESPTableMode::POLY;
        }
        else if (is_str_equal(mode, "table"))
        {
            esp->table_mode = ESPTableMode::TABLE;
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tesp_table_mode should be 'poly' or 'table'.");
        }
    }
    if (controller->Command_Exist(module_name, "esp_print_detail"))
    {
        esp->print_detail = controller->Get_Bool(
            module_name, "esp_print_detail", "Particle_Mesh::Initial");
    }
}

static void ESP_Upload_Float_Vector(float** device_ptr,
                                    const std::vector<float>& values)
{
    if (values.empty()) return;
    Device_Malloc_Safely((void**)device_ptr, sizeof(float) * values.size());
    deviceMemcpy(*device_ptr, values.data(), sizeof(float) * values.size(),
                 deviceMemcpyHostToDevice);
}

static void Allocate_ESP_PSWF_Buffers(Particle_Mesh* pme,
                                      const ESP_PSWF_Table& pswf_table)
{
    pme->ESP_near_grid_points =
        pswf_table.order * pswf_table.order * pswf_table.order;
    pme->ESP_window_table_size = pswf_table.order * pswf_table.table_points;
    pme->ESP_window_coeff_size =
        pswf_table.order * pswf_table.spread_poly_order;
    pme->ESP_scalar_table_size = pswf_table.table_points;
    pme->ESP_scalar_coeff_size = pswf_table.split_poly_order;

    ESP_Upload_Float_Vector(&pme->ESP_window_table,
                            pswf_table.spread_window_table);
    ESP_Upload_Float_Vector(&pme->ESP_window_derivative_table,
                            pswf_table.spread_window_derivative_table);
    ESP_Upload_Float_Vector(&pme->ESP_window_coeff,
                            pswf_table.spread_window_coeff);
    ESP_Upload_Float_Vector(&pme->ESP_window_derivative_coeff,
                            pswf_table.spread_window_derivative_coeff);
    ESP_Upload_Float_Vector(&pme->ESP_window_fourier_table,
                            pswf_table.spread_window_fourier_table);
    ESP_Upload_Float_Vector(&pme->ESP_window_fourier_coeff,
                            pswf_table.spread_window_fourier_coeff);
    ESP_Upload_Float_Vector(&pme->ESP_split_real_table,
                            pswf_table.split_real_table);
    ESP_Upload_Float_Vector(&pme->ESP_split_real_derivative_table,
                            pswf_table.split_real_derivative_table);
    ESP_Upload_Float_Vector(&pme->ESP_split_real_coeff,
                            pswf_table.split_real_coeff);
    ESP_Upload_Float_Vector(&pme->ESP_split_real_derivative_coeff,
                            pswf_table.split_real_derivative_coeff);
    ESP_Upload_Float_Vector(&pme->ESP_split_fourier_table,
                            pswf_table.split_fourier_table);
    ESP_Upload_Float_Vector(&pme->ESP_split_fourier_derivative_table,
                            pswf_table.split_fourier_derivative_table);
    ESP_Upload_Float_Vector(&pme->ESP_split_fourier_coeff,
                            pswf_table.split_fourier_coeff);
    ESP_Upload_Float_Vector(&pme->ESP_split_fourier_derivative_coeff,
                            pswf_table.split_fourier_derivative_coeff);
}

static float ESP_Eval_Host_Table(const std::vector<float>& table, float x)
{
    if (table.empty()) return 0.0f;
    if (x <= 0.0f) return table.front();
    if (x >= 1.0f) return table.back();
    float scaled = x * (table.size() - 1);
    int lower = (int)scaled;
    int upper = lower + 1;
    if (upper >= (int)table.size()) upper = table.size() - 1;
    float t = scaled - lower;
    return (1.0f - t) * table[lower] + t * table[upper];
}

static float ESP_Eval_Host_Poly(const std::vector<float>& coeff, float x)
{
    float y = 0.0f;
    for (int i = (int)coeff.size() - 1; i >= 0; i--)
    {
        y = y * x + coeff[i];
    }
    return y;
}

static float ESP_Eval_Host_Scalar(const std::vector<float>& table,
                                  const std::vector<float>& coeff,
                                  ESPTableMode table_mode, float x)
{
    if (x > 1.0f) return 0.0f;
    if (table_mode == ESPTableMode::POLY)
    {
        return ESP_Eval_Host_Poly(coeff, x);
    }
    return ESP_Eval_Host_Table(table, x);
}

static float ESP_Signed_Grid_Mode(int index, int n)
{
    float mode = index;
    if (index > n / 2) mode = index - n;
    return mode;
}

static float ESP_Spread_Fourier_Modulus(const ESP_PSWF_Table& pswf_table,
                                        ESPTableMode table_mode, int index,
                                        int n)
{
    float mode = fabsf(ESP_Signed_Grid_Mode(index, n));
    float arg =
        CONSTANT_Pi * pswf_table.order * mode / (n * pswf_table.c_spread);
    if (arg > 1.0f) return 1.0e30f;
    float window = ESP_Eval_Host_Scalar(pswf_table.spread_window_fourier_table,
                                        pswf_table.spread_window_fourier_coeff,
                                        table_mode, arg);
    float modulus = window * window;
    if (modulus < 1.0e-30f) modulus = 1.0e-30f;
    return modulus;
}

static void Build_ESP_BC(CONTROLLER* controller, Particle_Mesh* pme,
                         const ESP_PSWF_Table& pswf_table, LTMatrix3 rcell,
                         float volume)
{
    float* h_ESP_BC = (float*)malloc(sizeof(float) * pme->PME_Nfft);
    float* h_ESP_BC0 = (float*)malloc(sizeof(float) * pme->PME_Nfft);
    LTMatrix3* h_ESP_virial_BC =
        (LTMatrix3*)malloc(sizeof(LTMatrix3) * pme->PME_Nfft);
    if (h_ESP_BC == NULL || h_ESP_BC0 == NULL || h_ESP_virial_BC == NULL)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorMallocFailed, "Build_ESP_BC",
            "Reason:\n\tError occurs when malloc ESP_BC.");
    }

    std::vector<float> mod_x(pme->fftx);
    std::vector<float> mod_y(pme->ffty);
    std::vector<float> mod_z(pme->fftz);
    for (int i = 0; i < pme->fftx; i++)
    {
        mod_x[i] = ESP_Spread_Fourier_Modulus(pswf_table, pme->esp.table_mode,
                                              i, pme->fftx);
    }
    for (int i = 0; i < pme->ffty; i++)
    {
        mod_y[i] = ESP_Spread_Fourier_Modulus(pswf_table, pme->esp.table_mode,
                                              i, pme->ffty);
    }
    for (int i = 0; i < pme->fftz; i++)
    {
        mod_z[i] = ESP_Spread_Fourier_Modulus(pswf_table, pme->esp.table_mode,
                                              i, pme->fftz);
    }

    const float half_order = 0.5f * pswf_table.order;
    const float support_scale = half_order * half_order * half_order *
                                half_order * half_order * half_order;
    const float split_scale =
        2.0f * CONSTANT_Pi * pswf_table.cutoff / pswf_table.c_split;

    for (int kx = 0; kx < pme->fftx; kx++)
    {
        float kxrp = ESP_Signed_Grid_Mode(kx, pme->fftx);
        for (int ky = 0; ky < pme->ffty; ky++)
        {
            float kyrp = ESP_Signed_Grid_Mode(ky, pme->ffty);
            for (int kz = 0; kz <= pme->fftz / 2; kz++)
            {
                float kzrp = kz;
                int index = kx * pme->ffty * (pme->fftz / 2 + 1) +
                            ky * (pme->fftz / 2 + 1) + kz;
                VECTOR m = {kxrp, kyrp, kzrp};
                m = MultiplyTranspose(m, rcell);
                float msq = m * m;
                h_ESP_BC[index] = 0.0f;
                h_ESP_BC0[index] = 0.0f;
                h_ESP_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                if (kx + ky + kz == 0 || msq <= 0.0f)
                {
                    continue;
                }

                float split_arg = split_scale * sqrtf(msq);
                if (split_arg > 1.0f)
                {
                    continue;
                }
                float split_fourier =
                    0.5f * ESP_Eval_Host_Scalar(pswf_table.split_fourier_table,
                                                pswf_table.split_fourier_coeff,
                                                pme->esp.table_mode, split_arg);
                float split_fourier_derivative =
                    0.5f * ESP_Eval_Host_Scalar(
                               pswf_table.split_fourier_derivative_table,
                               pswf_table.split_fourier_derivative_coeff,
                               pme->esp.table_mode, split_arg);
                float deconvolution =
                    1.0f / (mod_x[kx] * mod_y[ky] * mod_z[kz] * support_scale);
            if (!ESP_Float_Is_Bounded(deconvolution, 1.0e12f))
                {
                    continue;
                }
                float bc = split_fourier * deconvolution /
                           (CONSTANT_Pi * msq * volume);
            if (!ESP_Float_Is_Bounded(bc, 1.0e6f))
                {
                    continue;
                }
                h_ESP_BC0[index] = deconvolution;
                h_ESP_BC[index] = bc;

                float derivative_ratio = 0.0f;
                if (fabsf(split_fourier) > 1.0e-30f)
                {
                    derivative_ratio = split_fourier_derivative / split_fourier;
                }
                float metric_factor = (2.0f - derivative_ratio) / msq;
                h_ESP_virial_BC[index].a11 =
                    0.5f * bc * (1.0f - metric_factor * m.x * m.x);
                h_ESP_virial_BC[index].a21 =
                    0.5f * bc * (0.0f - metric_factor * m.y * m.x);
                h_ESP_virial_BC[index].a22 =
                    0.5f * bc * (1.0f - metric_factor * m.y * m.y);
                h_ESP_virial_BC[index].a31 =
                    0.5f * bc * (0.0f - metric_factor * m.z * m.x);
                h_ESP_virial_BC[index].a32 =
                    0.5f * bc * (0.0f - metric_factor * m.z * m.y);
                h_ESP_virial_BC[index].a33 =
                    0.5f * bc * (1.0f - metric_factor * m.z * m.z);
            }
        }
    }

    Device_Malloc_Safely((void**)&pme->ESP_BC, sizeof(float) * pme->PME_Nfft);
    Device_Malloc_Safely((void**)&pme->ESP_BC0, sizeof(float) * pme->PME_Nfft);
    Device_Malloc_Safely((void**)&pme->ESP_Virial_BC,
                         sizeof(LTMatrix3) * pme->PME_Nfft);
    deviceMemcpy(pme->ESP_BC, h_ESP_BC, sizeof(float) * pme->PME_Nfft,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(pme->ESP_BC0, h_ESP_BC0, sizeof(float) * pme->PME_Nfft,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(pme->ESP_Virial_BC, h_ESP_virial_BC,
                 sizeof(LTMatrix3) * pme->PME_Nfft, deviceMemcpyHostToDevice);
    free(h_ESP_BC);
    free(h_ESP_BC0);
    free(h_ESP_virial_BC);
}

//--------Particle Mesh Ewald Method----------

void Particle_Mesh::Initial(CONTROLLER* controller, int atom_numbers,
                            LTMatrix3 cell, LTMatrix3 rcell, VECTOR box_length,
                            float cutoff,
                            int no_direct_interaction_virtual_atom_numbers,
                            const char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "PM");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }

    controller->printf("START INITIALIZING PME:\n");
    this->cutoff = cutoff;

    controller->printf("    PME backend library: %s\n", FFT_LIBRARY_NAME);

    tolerance = 0.00001;
    if (controller->Command_Exist(this->module_name, "Direct_Tolerance"))
    {
        controller->Check_Float(this->module_name, "Direct_Tolerance",
                                "Particle_Mesh::Initial");
        tolerance =
            atof(controller->Command(this->module_name, "Direct_Tolerance"));
    }

    backend = ParticleMeshBackend::PME;
    if (controller->Command_Exist(this->module_name, "backend"))
    {
        const char* backend_name =
            controller->Command(this->module_name, "backend");
        if (is_str_equal(backend_name, "pme"))
        {
            backend = ParticleMeshBackend::PME;
        }
        else if (is_str_equal(backend_name, "esp"))
        {
            backend = ParticleMeshBackend::ESP;
        }
        else
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
                "Reason:\n\tPM backend should be 'pme' or 'esp'.");
        }
    }
    Parse_ESP_Parameters(controller, this->module_name, &esp, tolerance,
                         cutoff);
    controller->printf("    particle mesh backend: %s\n",
                       Particle_Mesh_Backend_Name(backend));
    const bool pm_grid_spacing_explicit =
        controller->Command_Exist(this->module_name, "grid_spacing");
    const bool esp_order_explicit = esp.order > 0;
    const bool esp_grid_spacing_explicit = esp.grid_spacing > 0.0f;

    if (CONTROLLER::PP_MPI_size == 1)
    {
        exclude_factor = 1.0f;
    }
    else
    {
        exclude_factor = 0.5f;
    }

    fftx = -1;
    ffty = -1;
    fftz = -1;
    if (controller->Command_Exist(this->module_name, "fftx"))
    {
        controller->Check_Int(this->module_name, "fftx",
                              "Particle_Mesh::Initial");
        fftx = atoi(controller->Command(this->module_name, "fftx"));
    }
    if (controller->Command_Exist(this->module_name, "ffty"))
    {
        controller->Check_Int(this->module_name, "ffty",
                              "Particle_Mesh::Initial");
        ffty = atoi(controller->Command(this->module_name, "ffty"));
    }
    if (controller->Command_Exist(this->module_name, "fftz"))
    {
        controller->Check_Int(this->module_name, "fftz",
                              "Particle_Mesh::Initial");
        fftz = atoi(controller->Command(this->module_name, "fftz"));
    }

    PM_MPI_size = 0;
    if (controller->Command_Exist(this->module_name, "MPI_size"))
    {
        controller->Check_Int(this->module_name, "MPI_size",
                              "Particle_Mesh::Initial");
        PM_MPI_size = atoi(controller->Command(this->module_name, "MPI_size"));
    }
    else
    {
        PM_MPI_size = controller->PM_MPI_size;
    }
    if (!PM_MPI_size)
    {
        controller->printf("PM RECI NOT INITIALIZED");
    }
    // Multi-process PME/ESP is not enabled in the current implementation.
    if (PM_MPI_size > 1)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
            "Reason:\n\t Multi-process PME is not supported yet.");
    }

    this->atom_numbers = atom_numbers;
    this->max_atom_numbers = atom_numbers;
    Device_Malloc_Safely((void**)&num_ghost_dir_id,
                         sizeof(int) * max_atom_numbers * 6);

    float volume = cell.a11 * cell.a22 * cell.a33;
    ESP_Default_Selection esp_defaults;
    const bool fft_explicit_before_auto =
        fftx >= 0 || ffty >= 0 || fftz >= 0;
    const bool esp_auto_grid =
        backend == ParticleMeshBackend::ESP && !fft_explicit_before_auto &&
        !esp_grid_spacing_explicit && !pm_grid_spacing_explicit;
    if (backend == ParticleMeshBackend::ESP)
    {
        esp_defaults = Build_ESP_Default_Selection(esp.tolerance, cutoff,
                                                   box_length);
        if (esp_auto_grid)
        {
            fftx = esp_defaults.fftx;
            ffty = esp_defaults.ffty;
            fftz = esp_defaults.fftz;
            esp.grid_spacing = esp_defaults.target_grid_spacing;
        }
    }

    float grid_spacing = 1;
    if (pm_grid_spacing_explicit)
    {
        controller->Check_Float(this->module_name, "grid_spacing",
                                "Particle_Mesh::Initial");
        grid_spacing =
            atof(controller->Command(this->module_name, "grid_spacing"));
    }
    if (backend == ParticleMeshBackend::ESP && esp.grid_spacing > 0.0f)
    {
        grid_spacing = esp.grid_spacing;
    }
    controller->printf("    grid_spacing: %f Angstrom\n", grid_spacing);
    if (fftx < 0) fftx = Get_Fft_Patameter(box_length.x / grid_spacing);

    if (ffty < 0) ffty = Get_Fft_Patameter(box_length.y / grid_spacing);

    if (fftz < 0) fftz = Get_Fft_Patameter(box_length.z / grid_spacing);

    const float esp_effective_grid_spacing =
        backend == ParticleMeshBackend::ESP
            ? std::min(box_length.x / fftx,
                       std::min(box_length.y / ffty, box_length.z / fftz))
            : 0.0f;
    const bool esp_auto_order =
        backend == ParticleMeshBackend::ESP && !esp_order_explicit;
    if (esp_auto_order)
    {
        esp.order = esp_defaults.order;
    }
    const bool esp_auto_bandlimit =
        backend == ParticleMeshBackend::ESP && (esp_auto_grid || esp_auto_order);
    if (esp_auto_bandlimit)
    {
        esp.c_split = esp_defaults.c_split;
        esp.c_spread = esp_defaults.c_window;
    }

    controller->printf("    fftx: %d\n", fftx);
    controller->printf("    ffty: %d\n", ffty);
    controller->printf("    fftz: %d\n", fftz);

    PME_Nall = fftx * ffty * fftz;
    PME_Nin = ffty * fftz;
    PME_Nfft = fftx * ffty * (fftz / 2 + 1);

    beta = Get_Beta(cutoff, tolerance);
    controller->printf("    beta: %f\n", beta);

    neutralizing_factor = -0.5 * CONSTANT_Pi / (beta * beta * volume);
    Device_Malloc_Safely((void**)&charge_sum, sizeof(float));
    Device_Malloc_Safely((void**)&charge_square, sizeof(float) * atom_numbers);

    int i, kx, ky, kz, index;
    FFT_RESULT errP1, errP2;
    update_interval = 1;
    if (controller->Command_Exist("PME", "update_interval"))
    {
        controller->Check_Int("PME", "update_interval",
                              "Particle_Mesh::Initial");
        update_interval = atoi(controller->Command("PME", "update_interval"));
    }
    Device_Malloc_Safely((void**)&force_backup, sizeof(VECTOR) * atom_numbers);
    deviceMemset(force_backup, 0, sizeof(VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&PME_uxyz,
                         sizeof(UNSIGNED_INT_VECTOR) * atom_numbers);
    Device_Malloc_Safely((void**)&PME_frxyz, sizeof(VECTOR) * atom_numbers);
    Reset_List((int*)PME_uxyz, 1 << 30, 3 * atom_numbers);

    Device_Malloc_Safely((void**)&PME_Q, sizeof(float) * PME_Nall);
    Device_Malloc_Safely((void**)&PME_FQ, sizeof(FFT_COMPLEX) * PME_Nfft);
    Device_Malloc_Safely((void**)&PME_FBCFQ, sizeof(float) * PME_Nall);

    Device_Malloc_Safely((void**)&PME_atom_near,
                         sizeof(int) * 64 * atom_numbers);
    deviceMemset(PME_atom_near, 0, sizeof(int) * 64 * atom_numbers);

    FFT_SIZE_t n3d[3] = {fftx, ffty, fftz};
    errP1 =
        SPONGE_FFT_WRAPPER::Make_FFT_Plan(&PME_plan_r2c, 1, 3, n3d, FFT_R2C);
    errP2 =
        SPONGE_FFT_WRAPPER::Make_FFT_Plan(&PME_plan_c2r, 1, 3, n3d, FFT_C2R);
    if (errP1 != FFT_SUCCESS || errP2 != FFT_SUCCESS)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand, "Particle_Mesh::Initial",
            "Reason:\n\tError occurs when create fft plan of PME");
    }

    Device_Malloc_And_Copy_Safely((void**)&d_reciprocal_ene, &reciprocal_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_self_ene, &self_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_direct_ene, &direct_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_correction_ene, &correction_ene,
                                  sizeof(float));
    Device_Malloc_And_Copy_Safely((void**)&d_ee_ene, &ee_ene, sizeof(float));
    Device_Malloc_Safely((void**)&d_direct_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&d_correction_atom_energy,
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_id_l_g, sizeof(int) * atom_numbers);
    Device_Malloc_Safely((void**)&atom_id_g_l, sizeof(int) * atom_numbers);
    Device_Malloc_Safely(
        (void**)&g_crd,
        sizeof(VECTOR) *
            (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    Device_Malloc_Safely(
        (void**)&g_frc,
        sizeof(VECTOR) *
            (atom_numbers + no_direct_interaction_virtual_atom_numbers));
    deviceMemset(atom_id_l_g, 0, sizeof(int) * atom_numbers);
    deviceMemset(atom_id_g_l, 0, sizeof(int) * atom_numbers);
    deviceMemset(g_crd, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    deviceMemset(g_frc, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
    deviceMemset(d_direct_atom_energy, 0, sizeof(float) * atom_numbers);
    deviceMemset(d_correction_atom_energy, 0, sizeof(float) * atom_numbers);

    calculate_reciprocal_part = true;
    if (controller->Command_Exist("PME", "calculate_reciprocal_part"))
    {
        calculate_reciprocal_part = controller->Get_Bool(
            "PME", "calculate_reciprocal_part", "Particle_Mesh::Initial");
    }
    calculate_excluded_part = true;
    if (controller->Command_Exist("PME", "calculate_excluded_part"))
    {
        calculate_excluded_part = controller->Get_Bool(
            "PME", "calculate_excluded_part", "Particle_Mesh::Initial");
    }
    bool use_pmc_iz = false;
    if (controller->Command_Exist("PME", "replaced_by_PMC_IZ"))
    {
        use_pmc_iz = controller->Get_Bool("PME", "replaced_by_PMC_IZ",
                                          "Particle_Mesh::Initial");
    }
    if (backend == ParticleMeshBackend::ESP)
    {
        neutralizing_factor = 0.0f;
        if (PM_MPI_size > 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorNotImplemented, "Particle_Mesh::Initial",
                "Reason:\n\tESP does not support multi-process PME yet.");
        }
        if (use_pmc_iz)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorConflictingCommand, "Particle_Mesh::Initial",
                "Reason:\n\tPM backend 'esp' conflicts with "
                "PME.replaced_by_PMC_IZ.");
        }
        controller->printf("    ESP tolerance: %e\n", esp.tolerance);
        controller->printf("    ESP order: %d\n", esp.order);
        controller->printf("    ESP parameter mode: %s\n",
                           ESP_Parameter_Mode_Name(esp.parameter_mode));
        controller->printf("    ESP table mode: %s\n",
                           ESP_Table_Mode_Name(esp.table_mode));
        controller->printf("    ESP table points: %d\n", esp.table_points);
        controller->printf("    ESP default c_split estimate: %.8e\n",
                           esp_defaults.c_split);
        controller->printf("    ESP default c_window estimate: %.8e\n",
                           esp_defaults.c_window);
        controller->printf("    ESP default alpha estimate: %.8e\n",
                           esp_defaults.alpha);
        if (esp.grid_spacing > 0.0f)
        {
            controller->printf("    ESP grid_spacing override: %f Angstrom\n",
                               esp.grid_spacing);
        }
        if (esp_auto_grid)
        {
            controller->printf(
                "    ESP auto grid target spacing: %f Angstrom\n",
                esp_defaults.target_grid_spacing);
            controller->printf(
                "    ESP auto effective spacing: %f Angstrom\n",
                esp_effective_grid_spacing);
        }
        if (esp_auto_order)
        {
            controller->printf("    ESP auto order from tolerance/grid: %d\n",
                               esp.order);
        }
        ESP_PSWF_Table esp_pswf;
        try
        {
            esp_pswf = Build_ESP_PSWF_Table(esp);
        }
        catch (const std::exception& e)
        {
            controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                           "Particle_Mesh::Initial", e.what());
        }
        esp.order = esp_pswf.order;
        esp.table_points = esp_pswf.table_points;
        esp.c_spread = esp_pswf.c_spread;
        esp.c_split = esp_pswf.c_split;
        esp.c0_split = esp_pswf.c0_split;
        esp.psi0_split = esp_pswf.psi0_split;
        esp.lambda_split = esp_pswf.lambda_split;
        esp.lambda_spread = esp_pswf.lambda_spread;
        const float esp_raw_self_energy_coeff = esp_pswf.self_energy_coeff;
        esp.self_energy_coeff = beta / sqrtf(CONSTANT_Pi);
        esp.max_window_table_error = esp_pswf.max_window_table_error;
        esp.max_split_table_error = esp_pswf.max_split_table_error;
        esp.max_window_poly_error = esp_pswf.max_window_poly_error;
        esp.max_split_poly_error = esp_pswf.max_split_poly_error;
        esp.spread_poly_order = esp_pswf.spread_poly_order;
        esp.split_poly_order = esp_pswf.split_poly_order;
        Allocate_ESP_PSWF_Buffers(this, esp_pswf);
        Build_ESP_BC(controller, this, esp_pswf, rcell, volume);
        controller->printf("    ESP c_spread: %.8e\n", esp.c_spread);
        controller->printf("    ESP c_split: %.8e\n", esp.c_split);
        controller->printf("    ESP c0_split: %.8e\n", esp.c0_split);
        controller->printf("    ESP psi0_split: %.8e\n", esp.psi0_split);
        controller->printf("    ESP lambda_split: %.8e\n", esp.lambda_split);
        controller->printf("    ESP lambda_spread: %.8e\n", esp.lambda_spread);
        controller->printf("    ESP raw self_energy_coeff: %.8e\n",
                           esp_raw_self_energy_coeff);
        controller->printf("    ESP self_energy_coeff: %.8e\n",
                           esp.self_energy_coeff);
        controller->printf("    ESP near grid points per atom: %d\n",
                           ESP_near_grid_points);
        if (esp.print_detail)
        {
            controller->printf("    ESP spread poly order: %d\n",
                               esp.spread_poly_order);
            controller->printf("    ESP split poly order: %d\n",
                               esp.split_poly_order);
            controller->printf("    ESP window table floats: %d\n",
                               ESP_window_table_size);
            controller->printf("    ESP window coeff floats: %d\n",
                               ESP_window_coeff_size);
            controller->printf("    ESP scalar table floats: %d\n",
                               ESP_scalar_table_size);
            controller->printf("    ESP scalar coeff floats: %d\n",
                               ESP_scalar_coeff_size);
            controller->printf("    ESP influence coefficients: %d\n",
                               PME_Nfft);
            controller->printf("    ESP window table max error: %.8e\n",
                               esp.max_window_table_error);
            controller->printf("    ESP split table max error: %.8e\n",
                               esp.max_split_table_error);
            controller->printf("    ESP window poly max error: %.8e\n",
                               esp.max_window_poly_error);
            controller->printf("    ESP split poly max error: %.8e\n",
                               esp.max_split_poly_error);
        }
        controller->printf(
            "    WARNING: ESP backend is experimental; use current validation "
            "fixtures before production runs.\n");
    }

    // 计算B-Spline修正系数 * 泊松算子因子， 用于倒空间乘法
    if (calculate_reciprocal_part)
    {
        if (use_pmc_iz)
        {
            controller->printf("    PMC-IZ will be used instead of PME\n");
            if (controller->Command_Choice("mode", "npt"))
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorConflictingCommand, "Particle_Mesh::Initial",
                    "Reason:\n\tPMC-IZ can not be used in NPT mode");
            }
            Build_PMC_IZ_BC(
                controller, fftx, ffty, fftz, PME_Nfft, PME_Nall, PME_Nin,
                1.0f / box_length.x / box_length.x,
                1.0f / box_length.y / box_length.y, box_length.z / fftz, beta,
                CONSTANT_Pi / PME_Nall / box_length.x / box_length.y, &PME_BC);
        }
        else
        {
            float *B1 = NULL, *B2 = NULL, *B3 = NULL, *h_PME_BC = NULL,
                  *h_PME_BC0 = NULL;
            LTMatrix3* h_PME_virial_BC = NULL;
            B1 = (float*)malloc(sizeof(float) * fftx);
            ;
            B2 = (float*)malloc(sizeof(float) * ffty);
            B3 = (float*)malloc(sizeof(float) * fftz);
            h_PME_BC0 = (float*)malloc(sizeof(float) * PME_Nfft);
            h_PME_BC = (float*)malloc(sizeof(float) * PME_Nfft);
            h_PME_virial_BC = (LTMatrix3*)malloc(sizeof(LTMatrix3) * PME_Nfft);
            if (B1 == NULL || B2 == NULL || B3 == NULL || h_PME_BC0 == NULL ||
                h_PME_BC == NULL)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorMallocFailed, "Particle_Mesh::Initial",
                    "Reason:\n\tError occurs when malloc PME_BC of PME");
            }
            for (kx = 0; kx < fftx; kx++)
            {
                B1[kx] = getb(kx, fftx, 4);
            }

            for (ky = 0; ky < ffty; ky++)
            {
                B2[ky] = getb(ky, ffty, 4);
            }

            for (kz = 0; kz < fftz; kz++)
            {
                B3[kz] = getb(kz, fftz, 4);
            }

            float kxrp, kyrp, kzrp;
            float mprefactor = PI * PI / beta / beta;
            float msq;
            VECTOR m;
            for (kx = 0; kx < fftx; kx++)
            {
                kxrp = kx;
                if (kx > fftx / 2) kxrp = kx - fftx;
                for (ky = 0; ky < ffty; ky++)
                {
                    kyrp = ky;
                    if (ky > ffty / 2) kyrp = ky - ffty;
                    for (kz = 0; kz <= fftz / 2; kz++)
                    {
                        kzrp = kz;
                        m = {kxrp, kyrp, kzrp};
                        m = MultiplyTranspose(m, rcell);
                        msq = m * m;

                        index = kx * ffty * (fftz / 2 + 1) +
                                ky * (fftz / 2 + 1) + kz;

                        if (kx + ky + kz == 0)
                        {
                            h_PME_BC[index] = 0;
                            h_PME_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                        }
                        else
                        {
                            h_PME_BC[index] = (float)1.0 / PI / msq *
                                              exp(-mprefactor * msq) / volume;
                            h_PME_virial_BC[index].a11 =
                                1 -
                                2 / msq * (1 + mprefactor * msq) * m.x * m.x;
                            h_PME_virial_BC[index].a21 =
                                0 -
                                2 / msq * (1 + mprefactor * msq) * m.y * m.x;
                            h_PME_virial_BC[index].a22 =
                                1 -
                                2 / msq * (1 + mprefactor * msq) * m.y * m.y;
                            h_PME_virial_BC[index].a31 =
                                0 -
                                2 / msq * (1 + mprefactor * msq) * m.z * m.x;
                            h_PME_virial_BC[index].a32 =
                                0 -
                                2 / msq * (1 + mprefactor * msq) * m.z * m.y;
                            h_PME_virial_BC[index].a33 =
                                1 -
                                2 / msq * (1 + mprefactor * msq) * m.z * m.z;
                        }
                        h_PME_BC0[index] = B1[kx] * B2[ky] * B3[kz];
                        h_PME_BC[index] *= h_PME_BC0[index];
                        h_PME_virial_BC[index] =
                            0.5f * h_PME_BC[index] * h_PME_virial_BC[index];
                    }
                }
            }

            Device_Malloc_Safely((void**)&PME_BC, sizeof(float) * PME_Nfft);
            Device_Malloc_Safely((void**)&PME_BC0, sizeof(float) * PME_Nfft);
            Device_Malloc_Safely((void**)&PME_Virial_BC,
                                 sizeof(LTMatrix3) * PME_Nfft);
            deviceMemcpy(PME_BC, h_PME_BC, sizeof(float) * PME_Nfft,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(PME_BC0, h_PME_BC0, sizeof(float) * PME_Nfft,
                         deviceMemcpyHostToDevice);
            deviceMemcpy(PME_Virial_BC, h_PME_virial_BC,
                         sizeof(LTMatrix3) * PME_Nfft,
                         deviceMemcpyHostToDevice);
            free(B1);
            free(B2);
            free(B3);
            free(h_PME_BC0);
            free(h_PME_BC);
            free(h_PME_virial_BC);
        }
    }
    is_initialized = 1;
    if (is_initialized && !is_controller_printf_initialized)
    {
        controller->Step_Print_Initial(this->module_name, "%.2f");
        if (controller->Command_Exist(this->module_name, "print_detail"))
        {
            print_detail = controller->Get_Bool(
                this->module_name, "print_detail", "Particle_Mesh::Initial");
            if (print_detail)
            {
                controller->Step_Print_Initial("PM_direct", "%.2f");
                controller->Step_Print_Initial("PM_reciprocal", "%.2f");
                controller->Step_Print_Initial("PM_self", "%.2f");
                controller->Step_Print_Initial("PM_correction", "%.2f");
            }
        }
        is_controller_printf_initialized = 1;
        controller->printf("    structure last modify date is %d\n",
                           last_modify_date);
    }
    controller->printf("END INITIALIZING PME\n\n");
}

void Particle_Mesh::Clear()
{
    if (is_initialized)
    {
        is_initialized = 0;
        Free_Single_Device_Pointer((void**)&PME_uxyz);
        Free_Single_Device_Pointer((void**)&PME_frxyz);
        Free_Single_Device_Pointer((void**)&PME_Q);
        Free_Single_Device_Pointer((void**)&PME_FQ);
        Free_Single_Device_Pointer((void**)&PME_FBCFQ);
        Free_Single_Device_Pointer((void**)&PME_BC);
        Free_Single_Device_Pointer((void**)&PME_Virial_BC);
        Free_Single_Device_Pointer((void**)&PME_BC0);
        Free_Single_Device_Pointer((void**)&ESP_window_table);
        Free_Single_Device_Pointer((void**)&ESP_window_derivative_table);
        Free_Single_Device_Pointer((void**)&ESP_window_coeff);
        Free_Single_Device_Pointer((void**)&ESP_window_derivative_coeff);
        Free_Single_Device_Pointer((void**)&ESP_window_fourier_table);
        Free_Single_Device_Pointer((void**)&ESP_window_fourier_coeff);
        Free_Single_Device_Pointer((void**)&ESP_split_real_table);
        Free_Single_Device_Pointer((void**)&ESP_split_real_derivative_table);
        Free_Single_Device_Pointer((void**)&ESP_split_real_coeff);
        Free_Single_Device_Pointer((void**)&ESP_split_real_derivative_coeff);
        Free_Single_Device_Pointer((void**)&ESP_split_fourier_table);
        Free_Single_Device_Pointer((void**)&ESP_split_fourier_derivative_table);
        Free_Single_Device_Pointer((void**)&ESP_split_fourier_coeff);
        Free_Single_Device_Pointer((void**)&ESP_split_fourier_derivative_coeff);
        Free_Single_Device_Pointer((void**)&ESP_BC);
        Free_Single_Device_Pointer((void**)&ESP_BC0);
        Free_Single_Device_Pointer((void**)&ESP_Virial_BC);
        ESP_near_grid_points = 0;
        ESP_window_table_size = 0;
        ESP_window_coeff_size = 0;
        ESP_scalar_table_size = 0;
        ESP_scalar_coeff_size = 0;
        Free_Single_Device_Pointer((void**)&charge_sum);
        Free_Single_Device_Pointer((void**)&charge_square);
        Free_Single_Device_Pointer((void**)&num_ghost_dir_id);

        Free_Single_Device_Pointer((void**)&atom_id_l_g);
        Free_Single_Device_Pointer((void**)&atom_id_g_l);
        Free_Single_Device_Pointer((void**)&g_crd);
        Free_Single_Device_Pointer((void**)&g_frc);

        Free_Single_Device_Pointer((void**)&PME_atom_near);
        Free_Single_Device_Pointer((void**)&force_backup);

        SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&PME_plan_r2c);
        SPONGE_FFT_WRAPPER::Destroy_FFT_Plan(&PME_plan_c2r);

        Free_Host_And_Device_Pointer(NULL, (void**)&d_reciprocal_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_self_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_direct_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_correction_ene);
        Free_Host_And_Device_Pointer(NULL, (void**)&d_ee_ene);
        Free_Single_Device_Pointer((void**)&d_direct_atom_energy);
        Free_Single_Device_Pointer((void**)&d_correction_atom_energy);
    }
}

ESP_Direct_Parameters Particle_Mesh::Get_ESP_Direct_Parameters() const
{
    ESP_Direct_Parameters direct;
    direct.enabled = backend == ParticleMeshBackend::ESP;
    direct.table_points = esp.table_points;
    direct.split_poly_order = esp.split_poly_order;
    direct.use_polynomial_tables = esp.table_mode == ESPTableMode::POLY;
    direct.cutoff = esp.cutoff;
    direct.split_real_table = ESP_split_real_table;
    direct.split_real_derivative_table = ESP_split_real_derivative_table;
    direct.split_real_coeff = ESP_split_real_coeff;
    direct.split_real_derivative_coeff = ESP_split_real_derivative_coeff;
    return direct;
}

// 计算每个原子所在的网格点以及其周围64个网格点的索引
__global__ void PME_Atom_Near(const VECTOR* crd, int* PME_atom_near,
                              const int PME_Nin, const LTMatrix3 cell,
                              const LTMatrix3 rcell, const int atom_numbers,
                              const int fftx, const int ffty, const int fftz,
                              UNSIGNED_INT_VECTOR* PME_uxyz, VECTOR* PME_frxyz,
                              VECTOR* force_backup)
{
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        force_backup[atom] = {0.0f, 0.0f, 0.0f};
        UNSIGNED_INT_VECTOR* temp_uxyz = &PME_uxyz[atom];
        VECTOR frac_crd = crd[atom] * rcell;
        frac_crd = frac_crd - floorf(frac_crd);
        if (!ESP_Float_Is_Finite(frac_crd.x) ||
            !ESP_Float_Is_Finite(frac_crd.y) ||
            !ESP_Float_Is_Finite(frac_crd.z))
        {
            frac_crd = {0.0f, 0.0f, 0.0f};
        }
        int k, tempux, tempuy, tempuz;
        frac_crd.x *= fftx;
        tempux = (int)frac_crd.x;
        tempux = tempux < 0 ? 0 : (tempux < fftx ? tempux : fftx - 1);
        PME_frxyz[atom].x = frac_crd.x - tempux;
        PME_frxyz[atom].x = PME_frxyz[atom].x - floorf(PME_frxyz[atom].x);
        frac_crd.y *= ffty;
        tempuy = (int)frac_crd.y;
        tempuy = tempuy < 0 ? 0 : (tempuy < ffty ? tempuy : ffty - 1);
        PME_frxyz[atom].y = frac_crd.y - tempuy;
        PME_frxyz[atom].y = PME_frxyz[atom].y - floorf(PME_frxyz[atom].y);
        frac_crd.z *= fftz;
        tempuz = (int)frac_crd.z;
        tempuz = tempuz < 0 ? 0 : (tempuz < fftz ? tempuz : fftz - 1);
        PME_frxyz[atom].z = frac_crd.z - tempuz;
        PME_frxyz[atom].z = PME_frxyz[atom].z - floorf(PME_frxyz[atom].z);
        if (tempux != (*temp_uxyz).uint_x || tempuy != (*temp_uxyz).uint_y ||
            tempuz != (*temp_uxyz).uint_z)
        {
            (*temp_uxyz).uint_x = tempux;
            (*temp_uxyz).uint_y = tempuy;
            (*temp_uxyz).uint_z = tempuz;
            int* temp_near = PME_atom_near + atom * 64;
            int kx, ky, kz;
            for (k = 0; k < 64; k++)
            {
                kx = k / 16;
                ky = (k - 16 * kx) / 4;
                kz = k % 4;

                kx = tempux - kx;

                if (kx < 0) kx += fftx;
                if (kx >= fftx) kx -= fftx;
                ky = tempuy - ky;
                if (ky < 0) ky += ffty;
                if (ky >= ffty) ky -= ffty;
                kz = tempuz - kz;
                if (kz < 0) kz += fftz;
                if (kz >= fftz) kz -= fftz;
                temp_near[k] = kx * PME_Nin + ky * fftz + kz;
            }
        }
    }
}

// 将原子电荷分配到其周围的64个网格点上
__global__ void PME_Q_Spread(int* PME_atom_near, const float* charge,
                             const VECTOR* PME_frxyz, float* PME_Q,
                             const int atom_numbers, const int PME_Nall)
{
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        int k;
        float tempf, tempQ, tempf2;
        int* temp_near = PME_atom_near + atom * 64;
        VECTOR temp_frxyz = PME_frxyz[atom];
        float tempcharge = charge[atom];

        unsigned int kx;
#ifdef USE_GPU
        for (k = threadIdx.y; k < 64; k = k + blockDim.y)
#else
        for (k = 0; k < 64; k++)
#endif
        {
            kx = k / 16;
            tempf = temp_frxyz.x;
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];

            tempQ = tempcharge * tempf;

            kx = (k - kx * 16) / 4;
            tempf = temp_frxyz.y;
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];

            tempQ = tempQ * tempf;

            kx = k % 4;
            tempf = temp_frxyz.z;
            tempf2 = tempf * tempf;
            tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempQ = tempQ * tempf;

            int near_index = temp_near[k];
            if ((unsigned int)near_index < (unsigned int)PME_Nall)
            {
                atomicAdd(&PME_Q[near_index], tempQ);
            }
        }
    }
}

// 对FFT后的电荷密度进行修正
__global__ void PME_BCFQ(FFT_COMPLEX* PME_FQ, float* PME_BC, int PME_Nfft)
{
    SIMPLE_DEVICE_FOR(index, PME_Nfft)
    {
        float tempf = PME_BC[index];
        FFT_COMPLEX tempc = PME_FQ[index];
        REAL(PME_FQ[index]) = REAL(tempc) * tempf;
        IMAGINARY(PME_FQ[index]) = IMAGINARY(tempc) * tempf;
    }
}

static __global__ void ESP_Sanitize_Float_List(float* values,
                                               const int element_number)
{
    SIMPLE_DEVICE_FOR(index, element_number)
    {
        if (!ESP_Float_Is_Bounded(values[index], 1.0e4f))
        {
            values[index] = 0.0f;
        }
    }
}

static __global__ void ESP_Sanitize_Complex_List(FFT_COMPLEX* values,
                                                 const int element_number)
{
    SIMPLE_DEVICE_FOR(index, element_number)
    {
        if (!ESP_Float_Is_Bounded(REAL(values[index]), 1.0e8f))
        {
            REAL(values[index]) = 0.0f;
        }
        if (!ESP_Float_Is_Bounded(IMAGINARY(values[index]), 1.0e8f))
        {
            IMAGINARY(values[index]) = 0.0f;
        }
    }
}

static __global__ void ESP_Sanitize_Vector_List(VECTOR* values,
                                                const int element_number)
{
    SIMPLE_DEVICE_FOR(index, element_number)
    {
        if (!ESP_Float_Is_Bounded(values[index].x, 1.0e5f))
        {
            values[index].x = 0.0f;
        }
        if (!ESP_Float_Is_Bounded(values[index].y, 1.0e5f))
        {
            values[index].y = 0.0f;
        }
        if (!ESP_Float_Is_Bounded(values[index].z, 1.0e5f))
        {
            values[index].z = 0.0f;
        }
    }
}

// 计算每个原子受力
static __global__ void PME_Final(int* PME_atom_near, const float* charge,
                                 const float* PME_Q, VECTOR* force,
                                 const VECTOR* PME_frxyz, const LTMatrix3 rcell,
                                 const int fftx, const int ffty, const int fftz,
                                 const int atom_numbers, const int PME_Nall)
{
#ifdef GPU_ARCH_NAME
    int atom = blockDim.y * blockIdx.x + threadIdx.y;
    if (atom < atom_numbers)
#else
#pragma omp parallel for
    for (int atom = 0; atom < atom_numbers; atom++)
#endif
    {
        int k, kx;
        float tempdx, tempdy, tempdz, tempx, tempy, tempz, tempdQf;
        VECTOR tempdQ;
        float tempf, tempf2;
        float temp_charge = charge[atom];
        int* temp_near = PME_atom_near + atom * 64;
        VECTOR temp_frxyz = PME_frxyz[atom];
        VECTOR tempnv = {0, 0, 0};
#ifdef USE_GPU
        for (k = threadIdx.x; k < 64; k = k + blockDim.x)
#else
        for (k = 0; k < 64; k++)
#endif
        {
            int near_index = temp_near[k];
            if ((unsigned int)near_index >= (unsigned int)PME_Nall)
            {
                continue;
            }
            tempdQf = -PME_Q[near_index] * temp_charge;

            kx = k / 16;
            tempf = temp_frxyz.x;
            tempf2 = tempf * tempf;
            tempx = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempdx = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = (k - kx * 16) / 4;
            tempf = temp_frxyz.y;
            tempf2 = tempf * tempf;
            tempy = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempdy = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            kx = k % 4;
            tempf = temp_frxyz.z;
            tempf2 = tempf * tempf;
            tempz = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 +
                    PME_Mc[kx] * tempf + PME_Md[kx];
            tempdz = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

            tempdQ.x = tempdx * tempy * tempz * fftx;
            tempdQ.y = tempdy * tempx * tempz * ffty;
            tempdQ.z = tempdz * tempx * tempy * fftz;
            tempdQ = tempdQf * MultiplyTranspose(tempdQ, rcell);
            tempnv = tempnv + tempdQ;
        }
        Warp_Sum_To(force + atom, tempnv, 8);
    }
}

static __device__ __forceinline__ float ESP_Eval_Table_Window(
    const float* table, const int table_points, const int window_index, float x)
{
    if (x <= 0.0f) return table[window_index * table_points];
    if (x >= 1.0f) return table[window_index * table_points + table_points - 1];
    float scaled = x * (table_points - 1);
    int lower = (int)scaled;
    int upper = lower + 1;
    if (upper >= table_points) upper = table_points - 1;
    float t = scaled - lower;
    int offset = window_index * table_points;
    return (1.0f - t) * table[offset + lower] + t * table[offset + upper];
}

static __device__ __forceinline__ float ESP_Eval_Poly_Window(
    const float* coeff, const int poly_order, const int window_index, float x)
{
    int offset = window_index * poly_order;
    float y = 0.0f;
    for (int i = poly_order - 1; i >= 0; i--)
    {
        y = y * x + coeff[offset + i];
    }
    return y;
}

static __device__ __forceinline__ float ESP_Eval_Window(
    const float* table, const float* coeff, const int table_points,
    const int poly_order, const int use_poly, const int window_index, float x)
{
    if (use_poly)
    {
        return ESP_Eval_Poly_Window(coeff, poly_order, window_index, x);
    }
    return ESP_Eval_Table_Window(table, table_points, window_index, x);
}

static __device__ __forceinline__ void ESP_Decompose_Window_Index(
    int k, int order, int* kx, int* ky, int* kz)
{
    int order2 = order * order;
    *kx = k / order2;
    *ky = (k - (*kx) * order2) / order;
    *kz = k - (*kx) * order2 - (*ky) * order;
}

static __device__ __forceinline__ int ESP_Wrap_Subtract_Index(int base, int delta,
                                                              int n)
{
    int value = base - delta;
    if (value < 0) value += n;
    if (value >= n) value -= n;
    return value;
}

static __device__ __forceinline__ int ESP_Get_Grid_Index(
    const UNSIGNED_INT_VECTOR& uxyz, int kx, int ky, int kz, int fftx, int ffty,
    int fftz, int PME_Nin)
{
    int ix = ESP_Wrap_Subtract_Index(uxyz.uint_x, kx, fftx);
    int iy = ESP_Wrap_Subtract_Index(uxyz.uint_y, ky, ffty);
    int iz = ESP_Wrap_Subtract_Index(uxyz.uint_z, kz, fftz);
    return ix * PME_Nin + iy * fftz + iz;
}

static __device__ __forceinline__ void ESP_Fill_Window_Cache_1D(
    const float* window_table, const float* window_coeff, const int table_points,
    const int poly_order, const int use_poly, const int order,
    const VECTOR& temp_frxyz, float* wx, float* wy, float* wz)
{
    for (int i = 0; i < order; i++)
    {
        wx[i] = ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.x);
        wy[i] = ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.y);
        wz[i] = ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.z);
    }
}

static __device__ __forceinline__ void ESP_Fill_Window_Derivative_Cache_1D(
    const float* window_derivative_table,
    const float* window_derivative_coeff, const int table_points,
    const int poly_order, const int use_poly, const int order,
    const VECTOR& temp_frxyz, float* dwx, float* dwy, float* dwz)
{
    for (int i = 0; i < order; i++)
    {
        dwx[i] = ESP_Eval_Window(window_derivative_table,
                                 window_derivative_coeff, table_points,
                                 poly_order, use_poly, i, temp_frxyz.x);
        dwy[i] = ESP_Eval_Window(window_derivative_table,
                                 window_derivative_coeff, table_points,
                                 poly_order, use_poly, i, temp_frxyz.y);
        dwz[i] = ESP_Eval_Window(window_derivative_table,
                                 window_derivative_coeff, table_points,
                                 poly_order, use_poly, i, temp_frxyz.z);
    }
}

static __device__ __forceinline__ void ESP_Fill_Wrapped_Index_Cache_1D(
    const UNSIGNED_INT_VECTOR& uxyz, const int order, const int fftx,
    const int ffty, const int fftz, int* ix, int* iy, int* iz)
{
    for (int i = 0; i < order; i++)
    {
        ix[i] = ESP_Wrap_Subtract_Index(uxyz.uint_x, i, fftx);
        iy[i] = ESP_Wrap_Subtract_Index(uxyz.uint_y, i, ffty);
        iz[i] = ESP_Wrap_Subtract_Index(uxyz.uint_z, i, fftz);
    }
}

// ESP动态支持宽度版本：准备原子分数坐标、基网格点与备份力。
static __global__ void ESP_Atom_Near(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const int atom_numbers, const int fftx, const int ffty, const int fftz,
    UNSIGNED_INT_VECTOR* PME_uxyz, VECTOR* PME_frxyz, VECTOR* force_backup)
{
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        force_backup[atom] = {0.0f, 0.0f, 0.0f};
        UNSIGNED_INT_VECTOR* temp_uxyz = &PME_uxyz[atom];
        VECTOR frac_crd = crd[atom] * rcell;
        frac_crd = frac_crd - floorf(frac_crd);
        if (!ESP_Float_Is_Finite(frac_crd.x) ||
            !ESP_Float_Is_Finite(frac_crd.y) ||
            !ESP_Float_Is_Finite(frac_crd.z))
        {
            frac_crd = {0.0f, 0.0f, 0.0f};
        }

        frac_crd.x *= fftx;
        int tempux = (int)frac_crd.x;
        tempux = tempux < 0 ? 0 : (tempux < fftx ? tempux : fftx - 1);
        PME_frxyz[atom].x = frac_crd.x - tempux;
        PME_frxyz[atom].x = PME_frxyz[atom].x - floorf(PME_frxyz[atom].x);

        frac_crd.y *= ffty;
        int tempuy = (int)frac_crd.y;
        tempuy = tempuy < 0 ? 0 : (tempuy < ffty ? tempuy : ffty - 1);
        PME_frxyz[atom].y = frac_crd.y - tempuy;
        PME_frxyz[atom].y = PME_frxyz[atom].y - floorf(PME_frxyz[atom].y);

        frac_crd.z *= fftz;
        int tempuz = (int)frac_crd.z;
        tempuz = tempuz < 0 ? 0 : (tempuz < fftz ? tempuz : fftz - 1);
        PME_frxyz[atom].z = frac_crd.z - tempuz;
        PME_frxyz[atom].z = PME_frxyz[atom].z - floorf(PME_frxyz[atom].z);

        (*temp_uxyz).uint_x = tempux;
        (*temp_uxyz).uint_y = tempuy;
        (*temp_uxyz).uint_z = tempuz;
    }
}

// ESP/PSWF电荷分配：分离变量 W(x)W(y)W(z)，支持poly或table模式。
static __global__ void ESP_Q_Spread_Order5(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge,
    const VECTOR* PME_frxyz, float* PME_Q, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int fftx, const int ffty,
    const int fftz, const int table_points, const int poly_order,
    const int use_poly, const float* window_table, const float* window_coeff)
{
#ifdef GPU_ARCH_NAME
    int atoms_per_block = blockDim.x / ESP_GPU_SPREAD_LANES_PER_ATOM;
    int atom_local = threadIdx.x / ESP_GPU_SPREAD_LANES_PER_ATOM;
    int lane = threadIdx.x % ESP_GPU_SPREAD_LANES_PER_ATOM;
    int atom = atoms_per_block * blockIdx.x + atom_local;
    bool atom_valid = atom < atom_numbers;
    extern __shared__ unsigned char sm_buffer[];
    float* sm_wx = reinterpret_cast<float*>(sm_buffer);
    float* sm_wy = sm_wx + atoms_per_block * ESP_ORDER5;
    float* sm_wz = sm_wy + atoms_per_block * ESP_ORDER5;
    int* sm_ix = reinterpret_cast<int*>(sm_wz + atoms_per_block * ESP_ORDER5);
    int* sm_iy = sm_ix + atoms_per_block * ESP_ORDER5;
    int* sm_iz = sm_iy + atoms_per_block * ESP_ORDER5;
    if (atom_valid && lane < ESP_ORDER5)
    {
        VECTOR temp_frxyz = PME_frxyz[atom];
        UNSIGNED_INT_VECTOR temp_uxyz = PME_uxyz[atom];
        int cache_offset = atom_local * ESP_ORDER5 + lane;
        sm_wx[cache_offset] =
            ESP_Eval_Window(window_table, window_coeff, table_points,
                            poly_order, use_poly, lane, temp_frxyz.x);
        sm_wy[cache_offset] =
            ESP_Eval_Window(window_table, window_coeff, table_points,
                            poly_order, use_poly, lane, temp_frxyz.y);
        sm_wz[cache_offset] =
            ESP_Eval_Window(window_table, window_coeff, table_points,
                            poly_order, use_poly, lane, temp_frxyz.z);
        sm_ix[cache_offset] =
            ESP_Wrap_Subtract_Index(temp_uxyz.uint_x, lane, fftx);
        sm_iy[cache_offset] =
            ESP_Wrap_Subtract_Index(temp_uxyz.uint_y, lane, ffty);
        sm_iz[cache_offset] =
            ESP_Wrap_Subtract_Index(temp_uxyz.uint_z, lane, fftz);
    }
    deviceSyncWarp(FULL_MASK);
    if (atom_valid)
    {
        int cache_offset = atom_local * ESP_ORDER5;
        float tempcharge = charge[atom];
        for (int linear = lane; linear < ESP_ORDER5_SUPPORT;
             linear += ESP_GPU_SPREAD_LANES_PER_ATOM)
        {
            int kx = linear / 25;
            int rem = linear - kx * 25;
            int ky = rem / ESP_ORDER5;
            int kz = rem - ky * ESP_ORDER5;
            int near_index = sm_ix[cache_offset + kx] * PME_Nin +
                             sm_iy[cache_offset + ky] * fftz +
                             sm_iz[cache_offset + kz];
            if ((unsigned int)near_index < (unsigned int)PME_Nall)
            {
                atomicAdd(&PME_Q[near_index],
                          tempcharge * sm_wx[cache_offset + kx] *
                              sm_wy[cache_offset + ky] *
                              sm_wz[cache_offset + kz]);
            }
        }
    }
#else
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        float wx[ESP_ORDER5], wy[ESP_ORDER5], wz[ESP_ORDER5];
        int ix[ESP_ORDER5], iy[ESP_ORDER5], iz[ESP_ORDER5];
        VECTOR temp_frxyz = PME_frxyz[atom];
        UNSIGNED_INT_VECTOR temp_uxyz = PME_uxyz[atom];
        ESP_Fill_Window_Cache_1D(window_table, window_coeff, table_points,
                                 poly_order, use_poly, ESP_ORDER5, temp_frxyz,
                                 wx, wy, wz);
        ESP_Fill_Wrapped_Index_Cache_1D(temp_uxyz, ESP_ORDER5, fftx, ffty,
                                        fftz, ix, iy, iz);
        float tempcharge = charge[atom];
        for (int kx = 0; kx < ESP_ORDER5; kx++)
        {
            float charge_x = tempcharge * wx[kx];
            int base_x = ix[kx] * PME_Nin;
            for (int ky = 0; ky < ESP_ORDER5; ky++)
            {
                float charge_xy = charge_x * wy[ky];
                int base_xy = base_x + iy[ky] * fftz;
                for (int kz = 0; kz < ESP_ORDER5; kz++)
                {
                    int near_index = base_xy + iz[kz];
                    if ((unsigned int)near_index < (unsigned int)PME_Nall)
                    {
                        atomicAdd(&PME_Q[near_index], charge_xy * wz[kz]);
                    }
                }
            }
        }
    }
#endif
}

// ESP/PSWF电荷分配：分离变量 W(x)W(y)W(z)，支持poly或table模式。
static __global__ void ESP_Q_Spread(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge,
    const VECTOR* PME_frxyz, float* PME_Q, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int fftx, const int ffty,
    const int fftz, const int order, const int support, const int table_points,
    const int poly_order, const int use_poly, const float* window_table,
    const float* window_coeff)
{
#ifdef GPU_ARCH_NAME
    int atom = blockDim.x * blockIdx.x + threadIdx.x;
    bool atom_valid = atom < atom_numbers;
    extern __shared__ float sm_window_cache[];
    float* sm_wx = sm_window_cache;
    float* sm_wy = sm_wx + blockDim.x * order;
    float* sm_wz = sm_wy + blockDim.x * order;
    if (atom_valid)
    {
        VECTOR temp_frxyz = PME_frxyz[atom];
        int cache_offset = threadIdx.x * order;
        for (int i = threadIdx.y; i < order; i = i + blockDim.y)
        {
            sm_wx[cache_offset + i] =
                ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.x);
            sm_wy[cache_offset + i] =
                ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.y);
            sm_wz[cache_offset + i] =
                ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.z);
        }
    }
    __syncthreads();
    if (atom_valid)
#else
    for (int atom = 0; atom < atom_numbers; atom++)
#endif
    {
#ifndef GPU_ARCH_NAME
        std::vector<float> wx_cache(order), wy_cache(order), wz_cache(order);
        VECTOR temp_frxyz = PME_frxyz[atom];
        ESP_Fill_Window_Cache_1D(window_table, window_coeff, table_points,
                                 poly_order, use_poly, order, temp_frxyz,
                                 wx_cache.data(), wy_cache.data(),
                                 wz_cache.data());
#else
        int cache_offset = threadIdx.x * order;
#endif
        UNSIGNED_INT_VECTOR temp_uxyz = PME_uxyz[atom];
        float tempcharge = charge[atom];
#ifdef USE_GPU
        for (int k = threadIdx.y; k < support; k = k + blockDim.y)
#else
        for (int k = 0; k < support; k++)
#endif
        {
            int kx, ky, kz;
            ESP_Decompose_Window_Index(k, order, &kx, &ky, &kz);
#ifdef GPU_ARCH_NAME
            float wx = sm_wx[cache_offset + kx];
            float wy = sm_wy[cache_offset + ky];
            float wz = sm_wz[cache_offset + kz];
#else
            float wx = wx_cache[kx];
            float wy = wy_cache[ky];
            float wz = wz_cache[kz];
#endif
            int near_index = ESP_Get_Grid_Index(temp_uxyz, kx, ky, kz, fftx,
                                                ffty, fftz, PME_Nin);
            if ((unsigned int)near_index < (unsigned int)PME_Nall)
            {
                atomicAdd(&PME_Q[near_index], tempcharge * wx * wy * wz);
            }
        }
    }
}

// ESP/PSWF gather：使用 dW/dx, dW/dy, dW/dz 得到 reciprocal force。
static __global__ void ESP_Final_Order5(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge, const float* PME_Q,
    VECTOR* force, const VECTOR* PME_frxyz, const LTMatrix3 rcell,
    const int fftx, const int ffty, const int fftz, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int table_points,
    const int poly_order, const int use_poly, const float* window_table,
    const float* window_derivative_table, const float* window_coeff,
    const float* window_derivative_coeff)
{
#ifdef GPU_ARCH_NAME
    int atoms_per_block = blockDim.x / ESP_GPU_FINAL_LANES_PER_ATOM;
    int atom_local = threadIdx.x / ESP_GPU_FINAL_LANES_PER_ATOM;
    int lane = threadIdx.x % ESP_GPU_FINAL_LANES_PER_ATOM;
    int atom = atoms_per_block * blockIdx.x + atom_local;
    bool atom_valid = atom < atom_numbers;
    extern __shared__ unsigned char sm_buffer[];
    float* sm_wx = reinterpret_cast<float*>(sm_buffer);
    float* sm_wy = sm_wx + atoms_per_block * ESP_ORDER5;
    float* sm_wz = sm_wy + atoms_per_block * ESP_ORDER5;
    float* sm_dwx = sm_wz + atoms_per_block * ESP_ORDER5;
    float* sm_dwy = sm_dwx + atoms_per_block * ESP_ORDER5;
    float* sm_dwz = sm_dwy + atoms_per_block * ESP_ORDER5;
    int* sm_ix = reinterpret_cast<int*>(sm_dwz + atoms_per_block * ESP_ORDER5);
    int* sm_iy = sm_ix + atoms_per_block * ESP_ORDER5;
    int* sm_iz = sm_iy + atoms_per_block * ESP_ORDER5;
    if (atom_valid && lane < ESP_ORDER5)
    {
        VECTOR temp_frxyz = PME_frxyz[atom];
        UNSIGNED_INT_VECTOR temp_uxyz = PME_uxyz[atom];
        int cache_offset = atom_local * ESP_ORDER5 + lane;
        sm_wx[cache_offset] =
            ESP_Eval_Window(window_table, window_coeff, table_points,
                            poly_order, use_poly, lane, temp_frxyz.x);
        sm_wy[cache_offset] =
            ESP_Eval_Window(window_table, window_coeff, table_points,
                            poly_order, use_poly, lane, temp_frxyz.y);
        sm_wz[cache_offset] =
            ESP_Eval_Window(window_table, window_coeff, table_points,
                            poly_order, use_poly, lane, temp_frxyz.z);
        sm_dwx[cache_offset] =
            ESP_Eval_Window(window_derivative_table, window_derivative_coeff,
                            table_points, poly_order, use_poly, lane,
                            temp_frxyz.x);
        sm_dwy[cache_offset] =
            ESP_Eval_Window(window_derivative_table, window_derivative_coeff,
                            table_points, poly_order, use_poly, lane,
                            temp_frxyz.y);
        sm_dwz[cache_offset] =
            ESP_Eval_Window(window_derivative_table, window_derivative_coeff,
                            table_points, poly_order, use_poly, lane,
                            temp_frxyz.z);
        sm_ix[cache_offset] =
            ESP_Wrap_Subtract_Index(temp_uxyz.uint_x, lane, fftx);
        sm_iy[cache_offset] =
            ESP_Wrap_Subtract_Index(temp_uxyz.uint_y, lane, ffty);
        sm_iz[cache_offset] =
            ESP_Wrap_Subtract_Index(temp_uxyz.uint_z, lane, fftz);
    }
    deviceSyncWarp(FULL_MASK);
    if (atom_valid)
    {
        int cache_offset = atom_local * ESP_ORDER5;
        float temp_charge = charge[atom];
        VECTOR tempnv = {0, 0, 0};
        for (int linear = lane; linear < ESP_ORDER5_SUPPORT;
             linear += ESP_GPU_FINAL_LANES_PER_ATOM)
        {
            int kx = linear / 25;
            int rem = linear - kx * 25;
            int ky = rem / ESP_ORDER5;
            int kz = rem - ky * ESP_ORDER5;
            int near_index = sm_ix[cache_offset + kx] * PME_Nin +
                             sm_iy[cache_offset + ky] * fftz +
                             sm_iz[cache_offset + kz];
            if ((unsigned int)near_index >= (unsigned int)PME_Nall) continue;
            float wx = sm_wx[cache_offset + kx];
            float wy = sm_wy[cache_offset + ky];
            float wz = sm_wz[cache_offset + kz];
            float dwx = sm_dwx[cache_offset + kx];
            float dwy = sm_dwy[cache_offset + ky];
            float dwz = sm_dwz[cache_offset + kz];
            float tempdQf = -PME_Q[near_index] * temp_charge;
            VECTOR tempdQ;
            tempdQ.x = dwx * wy * wz * fftx;
            tempdQ.y = dwy * wx * wz * ffty;
            tempdQ.z = dwz * wx * wy * fftz;
            tempdQ = tempdQf * MultiplyTranspose(tempdQ, rcell);
            tempnv = tempnv + tempdQ;
        }
        for (int offset = ESP_GPU_FINAL_LANES_PER_ATOM >> 1; offset > 0;
             offset >>= 1)
        {
            tempnv.x +=
                deviceShflDown(FULL_MASK, tempnv.x, offset,
                               ESP_GPU_FINAL_LANES_PER_ATOM);
            tempnv.y +=
                deviceShflDown(FULL_MASK, tempnv.y, offset,
                               ESP_GPU_FINAL_LANES_PER_ATOM);
            tempnv.z +=
                deviceShflDown(FULL_MASK, tempnv.z, offset,
                               ESP_GPU_FINAL_LANES_PER_ATOM);
        }
        if (lane == 0)
        {
            atomicAdd(force + atom, tempnv);
        }
    }
#else
    SIMPLE_DEVICE_FOR(atom, atom_numbers)
    {
        float wx_cache[ESP_ORDER5], wy_cache[ESP_ORDER5], wz_cache[ESP_ORDER5];
        float dwx_cache[ESP_ORDER5], dwy_cache[ESP_ORDER5],
            dwz_cache[ESP_ORDER5];
        int ix_cache[ESP_ORDER5], iy_cache[ESP_ORDER5], iz_cache[ESP_ORDER5];
        VECTOR temp_frxyz = PME_frxyz[atom];
        UNSIGNED_INT_VECTOR temp_uxyz = PME_uxyz[atom];
        ESP_Fill_Window_Cache_1D(window_table, window_coeff, table_points,
                                 poly_order, use_poly, ESP_ORDER5, temp_frxyz,
                                 wx_cache, wy_cache, wz_cache);
        ESP_Fill_Window_Derivative_Cache_1D(
            window_derivative_table, window_derivative_coeff, table_points,
            poly_order, use_poly, ESP_ORDER5, temp_frxyz, dwx_cache,
            dwy_cache, dwz_cache);
        ESP_Fill_Wrapped_Index_Cache_1D(temp_uxyz, ESP_ORDER5, fftx, ffty,
                                        fftz, ix_cache, iy_cache, iz_cache);
        float temp_charge = charge[atom];
        VECTOR tempnv = {0, 0, 0};
        for (int kx = 0; kx < ESP_ORDER5; kx++)
        {
            int base_x = ix_cache[kx] * PME_Nin;
            float wx = wx_cache[kx];
            float dwx = dwx_cache[kx];
            for (int ky = 0; ky < ESP_ORDER5; ky++)
            {
                int base_xy = base_x + iy_cache[ky] * fftz;
                float wy = wy_cache[ky];
                float dwy = dwy_cache[ky];
                for (int kz = 0; kz < ESP_ORDER5; kz++)
                {
                    int near_index = base_xy + iz_cache[kz];
                    if ((unsigned int)near_index >= (unsigned int)PME_Nall)
                    {
                        continue;
                    }
                    float tempdQf = -PME_Q[near_index] * temp_charge;
                    VECTOR tempdQ;
                    tempdQ.x = dwx * wy * wz_cache[kz] * fftx;
                    tempdQ.y = dwy * wx * wz_cache[kz] * ffty;
                    tempdQ.z = dwz_cache[kz] * wx * wy * fftz;
                    tempdQ = tempdQf * MultiplyTranspose(tempdQ, rcell);
                    tempnv = tempnv + tempdQ;
                }
            }
        }
        atomicAdd(force + atom, tempnv);
    }
#endif
}

// ESP/PSWF gather：使用 dW/dx, dW/dy, dW/dz 得到 reciprocal force。
static __global__ void ESP_Final(
    const UNSIGNED_INT_VECTOR* PME_uxyz, const float* charge, const float* PME_Q,
    VECTOR* force, const VECTOR* PME_frxyz, const LTMatrix3 rcell,
    const int fftx, const int ffty, const int fftz, const int atom_numbers,
    const int PME_Nin, const int PME_Nall, const int order, const int support,
    const int table_points, const int poly_order, const int use_poly,
    const float* window_table, const float* window_derivative_table,
    const float* window_coeff, const float* window_derivative_coeff)
{
#ifdef GPU_ARCH_NAME
    int atom = blockDim.y * blockIdx.x + threadIdx.y;
    bool atom_valid = atom < atom_numbers;
    extern __shared__ float sm_window_cache[];
    float* sm_wx = sm_window_cache;
    float* sm_wy = sm_wx + blockDim.y * order;
    float* sm_wz = sm_wy + blockDim.y * order;
    float* sm_dwx = sm_wz + blockDim.y * order;
    float* sm_dwy = sm_dwx + blockDim.y * order;
    float* sm_dwz = sm_dwy + blockDim.y * order;
    if (atom_valid)
    {
        VECTOR temp_frxyz = PME_frxyz[atom];
        int cache_offset = threadIdx.y * order;
        for (int i = threadIdx.x; i < order; i = i + blockDim.x)
        {
            sm_wx[cache_offset + i] =
                ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.x);
            sm_wy[cache_offset + i] =
                ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.y);
            sm_wz[cache_offset + i] =
                ESP_Eval_Window(window_table, window_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.z);
            sm_dwx[cache_offset + i] =
                ESP_Eval_Window(window_derivative_table,
                                window_derivative_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.x);
            sm_dwy[cache_offset + i] =
                ESP_Eval_Window(window_derivative_table,
                                window_derivative_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.y);
            sm_dwz[cache_offset + i] =
                ESP_Eval_Window(window_derivative_table,
                                window_derivative_coeff, table_points,
                                poly_order, use_poly, i, temp_frxyz.z);
        }
    }
    __syncthreads();
    if (atom_valid)
#else
#pragma omp parallel for
    for (int atom = 0; atom < atom_numbers; atom++)
#endif
    {
#ifndef GPU_ARCH_NAME
        VECTOR temp_frxyz = PME_frxyz[atom];
#else
        VECTOR temp_frxyz = PME_frxyz[atom];
        int cache_offset = threadIdx.y * order;
#endif
        UNSIGNED_INT_VECTOR temp_uxyz = PME_uxyz[atom];
        float temp_charge = charge[atom];
        VECTOR tempnv = {0, 0, 0};
#ifndef GPU_ARCH_NAME
        if (order == ESP_ORDER5)
        {
            float wx_cache[ESP_ORDER5], wy_cache[ESP_ORDER5], wz_cache[ESP_ORDER5];
            float dwx_cache[ESP_ORDER5], dwy_cache[ESP_ORDER5],
                dwz_cache[ESP_ORDER5];
            int ix_cache[ESP_ORDER5], iy_cache[ESP_ORDER5], iz_cache[ESP_ORDER5];
            ESP_Fill_Window_Cache_1D(window_table, window_coeff, table_points,
                                     poly_order, use_poly, ESP_ORDER5,
                                     temp_frxyz, wx_cache, wy_cache, wz_cache);
            ESP_Fill_Window_Derivative_Cache_1D(
                window_derivative_table, window_derivative_coeff, table_points,
                poly_order, use_poly, ESP_ORDER5, temp_frxyz, dwx_cache,
                dwy_cache, dwz_cache);
            ESP_Fill_Wrapped_Index_Cache_1D(temp_uxyz, ESP_ORDER5, fftx, ffty,
                                            fftz, ix_cache, iy_cache, iz_cache);
            for (int kx = 0; kx < ESP_ORDER5; kx++)
            {
                int base_x = ix_cache[kx] * PME_Nin;
                float wx = wx_cache[kx];
                float dwx = dwx_cache[kx];
                for (int ky = 0; ky < ESP_ORDER5; ky++)
                {
                    int base_xy = base_x + iy_cache[ky] * fftz;
                    float wy = wy_cache[ky];
                    float dwy = dwy_cache[ky];
                    for (int kz = 0; kz < ESP_ORDER5; kz++)
                    {
                        int near_index = base_xy + iz_cache[kz];
                        if ((unsigned int)near_index >=
                            (unsigned int)PME_Nall)
                        {
                            continue;
                        }
                        float tempdQf = -PME_Q[near_index] * temp_charge;
                        VECTOR tempdQ;
                        tempdQ.x = dwx * wy * wz_cache[kz] * fftx;
                        tempdQ.y = dwy * wx * wz_cache[kz] * ffty;
                        tempdQ.z = dwz_cache[kz] * wx * wy * fftz;
                        tempdQ = tempdQf * MultiplyTranspose(tempdQ, rcell);
                        tempnv = tempnv + tempdQ;
                    }
                }
            }
            Warp_Sum_To(force + atom, tempnv, 8);
            continue;
        }
        std::vector<float> wx_cache(order), wy_cache(order), wz_cache(order);
        std::vector<float> dwx_cache(order), dwy_cache(order), dwz_cache(order);
        ESP_Fill_Window_Cache_1D(window_table, window_coeff, table_points,
                                 poly_order, use_poly, order, temp_frxyz,
                                 wx_cache.data(), wy_cache.data(),
                                 wz_cache.data());
        ESP_Fill_Window_Derivative_Cache_1D(
            window_derivative_table, window_derivative_coeff, table_points,
            poly_order, use_poly, order, temp_frxyz, dwx_cache.data(),
            dwy_cache.data(), dwz_cache.data());
#endif
#ifdef USE_GPU
        for (int k = threadIdx.x; k < support; k = k + blockDim.x)
#else
        for (int k = 0; k < support; k++)
#endif
        {
            int kx, ky, kz;
            ESP_Decompose_Window_Index(k, order, &kx, &ky, &kz);
            int near_index = ESP_Get_Grid_Index(temp_uxyz, kx, ky, kz, fftx,
                                                ffty, fftz, PME_Nin);
            if ((unsigned int)near_index >= (unsigned int)PME_Nall) continue;
#ifdef GPU_ARCH_NAME
            float wx = sm_wx[cache_offset + kx];
            float wy = sm_wy[cache_offset + ky];
            float wz = sm_wz[cache_offset + kz];
            float dwx = sm_dwx[cache_offset + kx];
            float dwy = sm_dwy[cache_offset + ky];
            float dwz = sm_dwz[cache_offset + kz];
#else
            float wx = wx_cache[kx];
            float wy = wy_cache[ky];
            float wz = wz_cache[kz];
            float dwx = dwx_cache[kx];
            float dwy = dwy_cache[ky];
            float dwz = dwz_cache[kz];
#endif
            float tempdQf = -PME_Q[near_index] * temp_charge;
            VECTOR tempdQ;
            tempdQ.x = dwx * wy * wz * fftx;
            tempdQ.y = dwy * wx * wz * ffty;
            tempdQ.z = dwz * wx * wy * fftz;
            tempdQ = tempdQf * MultiplyTranspose(tempdQ, rcell);
            tempnv = tempnv + tempdQ;
        }
        Warp_Sum_To(force + atom, tempnv, 8);
    }
}

// sum += list1 * list2
__global__ void PME_Energy_Product(const int element_number, const float* list1,
                                   const float* list2, float* sum)
{
#ifdef USE_GPU
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
#else
    sum[0] = 0;
#endif
    float lin = 0.0;
#ifdef USE_GPU
    for (int i = threadIdx.x; i < element_number; i = i + blockDim.x)
#else
#pragma omp parallel for reduction(+ : lin)
    for (int i = 0; i < element_number; i++)
#endif
    {
        lin = lin + list1[i] * list2[i];
    }
    atomicAdd(sum, lin);
}

static __global__ void ESP_Energy_Product(const int element_number,
                                          const float* list1,
                                          const float* list2, float* sum)
{
#ifdef USE_GPU
    if (threadIdx.x == 0)
    {
        sum[0] = 0.;
    }
    __syncthreads();
#else
    sum[0] = 0;
#endif
    double lin = 0.0;
#ifdef USE_GPU
    for (int i = threadIdx.x; i < element_number; i = i + blockDim.x)
#else
#pragma omp parallel for reduction(+ : lin)
    for (int i = 0; i < element_number; i++)
#endif
    {
        double product = 0.0;
        if (ESP_Float_Is_Bounded(list1[i], 1.0e6f) &&
            ESP_Float_Is_Bounded(list2[i], 1.0e6f))
        {
            product = static_cast<double>(list1[i]) * list2[i];
            if (product < 1.0e12 && product > -1.0e12)
            {
                lin += product;
            }
        }
    }
    if (lin > 1.0e20 || lin < -1.0e20)
    {
        lin = 0.0;
    }
    atomicAdd(sum, static_cast<float>(lin));
}

static __global__ void PME_Excluded_Force_With_Atom_Energy_Correction(
    const int atom_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const float pme_beta,
    const ESP_Direct_Parameters esp_direct, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    float* atom_ene, float* this_ene, LTMatrix3* atom_virial)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        int excluded_numbers = excluded_atom_numbers[atom_i];
        if (excluded_numbers > 0)
        {
            int list_start = excluded_list_start[atom_i];
            int list_end = list_start + excluded_numbers;
            int atom_j;

            float charge_i = charge[atom_i];
            float charge_j;
            float dr_abs;
            float beta_dr;

            VECTOR r1 = crd[atom_i], r2;
            VECTOR dr;
            float dr2;

            float frc_abs = 0.;
            VECTOR frc_lin;
            VECTOR frc_record = {0., 0., 0.};
            LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                atom_j = excluded_list[i];
                r2 = crd[atom_j];
                charge_j = charge[atom_j];

                dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
                dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                // 假设剔除表中的原子对距离总是小于cutoff的，正常体系

                dr_abs = sqrtf(dr2);
                if (esp_direct.enabled)
                {
                    frc_abs = ESP_Get_Excluded_Coulomb_Force(
                        charge_i * charge_j, dr_abs, esp_direct);
                }
                else
                {
                    beta_dr = pme_beta * dr_abs;
                    frc_abs = beta_dr * TWO_DIVIDED_BY_SQRT_PI *
                                  expf(-beta_dr * beta_dr) +
                              erfcf(beta_dr);
                    frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
                    frc_abs = -charge_i * charge_j * frc_abs;
                }
                frc_lin = frc_abs * dr;
                if (esp_direct.enabled)
                {
                    ene_lin += ESP_Get_Excluded_Coulomb_Energy(
                        charge_i * charge_j, dr_abs, esp_direct);
                }
                else
                {
                    ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;
                }
                frc_record = frc_record + frc_lin;
                atomicAdd(frc + atom_j, -frc_lin);
                virial_record =
                    virial_record - Get_Virial_From_Force_Dis(frc_lin, dr);
            }  // atom_j cycle
            atomicAdd(frc + atom_i, frc_record);
            atomicAdd(atom_virial + atom_i, virial_record);
            atomicAdd(atom_ene + atom_i, ene_lin);
            this_ene[atom_i] = ene_lin;
        }  // if need excluded
    }
}

static void Launch_PME_Excluded_Correction(
    Particle_Mesh* pm, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    float* atom_ene, LTMatrix3* atom_virial);

void Particle_Mesh::PME_Excluded_Force_With_Atom_Energy(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* charge, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    int need_energy, float* atom_ene, LTMatrix3* atom_virial)
{
    if (is_initialized && calculate_excluded_part)
    {
        if (need_energy)
            deviceMemset(d_correction_atom_energy, 0,
                         sizeof(float) * atom_numbers);
        if (CONTROLLER::MPI_rank != 0) return;
        Launch_PME_Excluded_Correction(this, crd, cell, rcell, charge,
                                       excluded_list_start, excluded_list,
                                       excluded_atom_numbers, frc, atom_ene,
                                       atom_virial);
    }
}

static __global__ void PME_Add_Energy_To_Potential(float* d_ene,
                                                   float* d_self_ene,
                                                   float* d_reciprocal_ene)
{
    d_ene[0] += d_self_ene[0] + d_reciprocal_ene[0];
}

static __global__ void device_add_force(const int atom_numbers,
                                        float update_interval, VECTOR* force,
                                        const VECTOR* force_backup)
{
    SIMPLE_DEVICE_FOR(atom_i, atom_numbers)
    {
        force[atom_i] = force[atom_i] + update_interval * force_backup[atom_i];
    }
}

static __global__ void PME_Sum_Virial(const int nfft,
                                      const LTMatrix3* virial_BC,
                                      const FFT_COMPLEX* FQ, LTMatrix3* virial,
                                      int fftz)
{
    LTMatrix3 vir = {0, 0, 0, 0, 0, 0};
#ifdef USE_GPU
    int tid = blockDim.x * blockIdx.x * blockDim.y + threadIdx.x * blockDim.y +
              threadIdx.y;
    for (int index = tid; index < nfft;
         index += blockDim.x * blockDim.y * gridDim.x)
    {
        int fftc = fftz / 2 + 1;
        int nz = index % fftc;
        float factor = (nz == 0 || nz == fftc - 1) ? 0.5f : 1.0f;
        FFT_COMPLEX FQ0 = FQ[index];
        LTMatrix3 vir0 =
            factor * (REAL(FQ0) * REAL(FQ0) + IMAGINARY(FQ0) * IMAGINARY(FQ0)) *
            virial_BC[index];
        vir = vir - vir0;
    }
#else
    float v11 = 0.0f, v21 = 0.0f, v22 = 0.0f;
    float v31 = 0.0f, v32 = 0.0f, v33 = 0.0f;
#pragma omp parallel for reduction(+ : v11, v21, v22, v31, v32, v33)
    for (int index = 0; index < nfft; index++)
    {
        int fftc = fftz / 2 + 1;
        int nz = index % fftc;
        float factor = (nz == 0 || nz == fftc - 1) ? 0.5f : 1.0f;
        FFT_COMPLEX FQ0 = FQ[index];
        LTMatrix3 vir0 =
            factor * (REAL(FQ0) * REAL(FQ0) + IMAGINARY(FQ0) * IMAGINARY(FQ0)) *
            virial_BC[index];
        v11 -= vir0.a11;
        v21 -= vir0.a21;
        v22 -= vir0.a22;
        v31 -= vir0.a31;
        v32 -= vir0.a32;
        v33 -= vir0.a33;
    }
    vir = {v11, v21, v22, v31, v32, v33};
#endif
    Warp_Sum_To(virial, vir, warpSize);
}

static __global__ void up_box_bc(int fftx, int ffty, int fftz, float* PME_BC,
                                 float* PME_BC0, LTMatrix3* PME_virial_BC,
                                 float mprefactor, LTMatrix3 rcell,
                                 float volume);
static __global__ void up_box_esp_bc(
    int fftx, int ffty, int fftz, float* ESP_BC, const float* ESP_BC0,
    LTMatrix3* ESP_virial_BC, LTMatrix3 rcell, float volume, float cutoff,
    float c_split, int table_points, int split_poly_order, int use_poly,
    const float* split_fourier_table, const float* split_fourier_coeff,
    const float* split_fourier_derivative_table,
    const float* split_fourier_derivative_coeff);
static void Scale_Positions_Device(const LTMatrix3 g, VECTOR* crd, float dt);

static void Launch_PME_Excluded_Correction(
    Particle_Mesh* pm, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    float* atom_ene, LTMatrix3* atom_virial)
{
    Launch_Device_Kernel(
        PME_Excluded_Force_With_Atom_Energy_Correction,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers, crd, cell,
        rcell, charge, pm->beta, pm->Get_ESP_Direct_Parameters(),
        excluded_list_start, excluded_list, excluded_atom_numbers, frc,
        atom_ene, pm->d_correction_atom_energy, atom_virial);
}

static void Run_ESP_Reciprocal_Force_Backend(
    Particle_Mesh* pm, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, VECTOR* force, int need_virial,
    LTMatrix3* d_virial, int step)
{
    if (step % pm->update_interval != 0)
    {
        return;
    }

    int use_poly = pm->esp.table_mode == ESPTableMode::POLY;
    deviceMemset(pm->PME_Q, 0, sizeof(float) * pm->PME_Nall);
    Launch_Device_Kernel(
        ESP_Atom_Near,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, crd, cell, rcell,
        pm->atom_numbers, pm->fftx, pm->ffty, pm->fftz, pm->PME_uxyz,
        pm->PME_frxyz, pm->force_backup);

    dim3 blockSize(1, 1, 1);
    if (pm->esp.order == ESP_ORDER5)
    {
        blockSize = {ESP_GPU_SPREAD_LANES_PER_ATOM *
                         ESP_GPU_SPREAD_ATOMS_PER_BLOCK,
                     1};
        size_t spread_shared_memory =
            ESP_GPU_SPREAD_ATOMS_PER_BLOCK * ESP_ORDER5 *
            (3 * sizeof(float) + 3 * sizeof(int));
        Launch_Device_Kernel(
            ESP_Q_Spread_Order5,
            (pm->atom_numbers + ESP_GPU_SPREAD_ATOMS_PER_BLOCK - 1) /
                ESP_GPU_SPREAD_ATOMS_PER_BLOCK,
            blockSize, spread_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_frxyz, pm->PME_Q, pm->atom_numbers, pm->PME_Nin,
            pm->PME_Nall, pm->fftx, pm->ffty, pm->fftz,
            pm->esp.table_points, pm->esp.spread_poly_order, use_poly,
            pm->ESP_window_table, pm->ESP_window_coeff);
    }
    else
    {
        int spread_threads_y = pm->ESP_near_grid_points;
        if (spread_threads_y > CONTROLLER::device_max_thread)
            spread_threads_y = CONTROLLER::device_max_thread;
        if (spread_threads_y < 1) spread_threads_y = 1;
        blockSize = {CONTROLLER::device_max_thread / spread_threads_y,
                     (unsigned int)spread_threads_y};
        if (blockSize.x < 1) blockSize.x = 1;
        size_t spread_shared_memory =
            sizeof(float) * blockSize.x * pm->esp.order * 3;
        Launch_Device_Kernel(
            ESP_Q_Spread, (pm->atom_numbers + blockSize.x - 1) / blockSize.x,
            blockSize, spread_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_frxyz, pm->PME_Q, pm->atom_numbers, pm->PME_Nin,
            pm->PME_Nall, pm->fftx, pm->ffty, pm->fftz, pm->esp.order,
            pm->ESP_near_grid_points, pm->esp.table_points,
            pm->esp.spread_poly_order, use_poly, pm->ESP_window_table,
            pm->ESP_window_coeff);
    }
    Launch_Device_Kernel(
        ESP_Sanitize_Float_List,
        (pm->PME_Nall + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_Q, pm->PME_Nall);

    SPONGE_FFT_WRAPPER::R2C(pm->PME_plan_r2c, pm->PME_Q, pm->PME_FQ);
    Launch_Device_Kernel(
        ESP_Sanitize_Complex_List,
        (pm->PME_Nfft + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FQ, pm->PME_Nfft);

    blockSize = {CONTROLLER::device_warp,
                 CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    if (need_virial)
    {
        Launch_Device_Kernel(
            PME_Sum_Virial,
            (pm->PME_Nfft + 4 * CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            blockSize, 0, NULL, pm->PME_Nfft, pm->ESP_Virial_BC, pm->PME_FQ,
            d_virial, pm->fftz);
    }

    Launch_Device_Kernel(
        PME_BCFQ,
        (pm->PME_Nfft + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FQ, pm->ESP_BC,
        pm->PME_Nfft);
    Launch_Device_Kernel(
        ESP_Sanitize_Complex_List,
        (pm->PME_Nfft + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FQ, pm->PME_Nfft);

    SPONGE_FFT_WRAPPER::C2R(pm->PME_plan_c2r, pm->PME_FQ, pm->PME_FBCFQ);
    Launch_Device_Kernel(
        ESP_Sanitize_Float_List,
        (pm->PME_Nall + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FBCFQ, pm->PME_Nall);

    if (pm->esp.order == ESP_ORDER5)
    {
        blockSize = {ESP_GPU_FINAL_LANES_PER_ATOM *
                         ESP_GPU_FINAL_ATOMS_PER_BLOCK,
                     1};
        size_t final_shared_memory =
            ESP_GPU_FINAL_ATOMS_PER_BLOCK * ESP_ORDER5 *
            (6 * sizeof(float) + 3 * sizeof(int));
        Launch_Device_Kernel(
            ESP_Final_Order5,
            (pm->atom_numbers + ESP_GPU_FINAL_ATOMS_PER_BLOCK - 1) /
                ESP_GPU_FINAL_ATOMS_PER_BLOCK,
            blockSize, final_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_FBCFQ, pm->force_backup, pm->PME_frxyz, rcell, pm->fftx,
            pm->ffty, pm->fftz, pm->atom_numbers, pm->PME_Nin, pm->PME_Nall,
            pm->esp.table_points, pm->esp.spread_poly_order, use_poly,
            pm->ESP_window_table, pm->ESP_window_derivative_table,
            pm->ESP_window_coeff, pm->ESP_window_derivative_coeff);
    }
    else
    {
        blockSize = {8, CONTROLLER::device_max_thread / 8};
        size_t final_shared_memory =
            sizeof(float) * blockSize.y * pm->esp.order * 6;
        Launch_Device_Kernel(
            ESP_Final, (pm->atom_numbers + blockSize.y - 1) / blockSize.y,
            blockSize, final_shared_memory, NULL, pm->PME_uxyz, charge,
            pm->PME_FBCFQ,
            pm->force_backup, pm->PME_frxyz, rcell, pm->fftx, pm->ffty,
            pm->fftz, pm->atom_numbers, pm->PME_Nin, pm->PME_Nall,
            pm->esp.order, pm->ESP_near_grid_points, pm->esp.table_points,
            pm->esp.spread_poly_order, use_poly, pm->ESP_window_table,
            pm->ESP_window_derivative_table, pm->ESP_window_coeff,
            pm->ESP_window_derivative_coeff);
    }
    Launch_Device_Kernel(
        ESP_Sanitize_Vector_List,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->force_backup,
        pm->atom_numbers);

    Launch_Device_Kernel(
        device_add_force,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers,
        pm->update_interval, force, pm->force_backup);
    Launch_Device_Kernel(
        ESP_Sanitize_Vector_List,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, force, pm->atom_numbers);
}

static void Run_ESP_Reciprocal_Energy_Backend(Particle_Mesh* pm,
                                              const float* charge,
                                              float* d_potential)
{
    Launch_Device_Kernel(ESP_Energy_Product, 1, CONTROLLER::device_max_thread,
                         0, NULL, pm->PME_Nall, pm->PME_Q, pm->PME_FBCFQ,
                         pm->d_reciprocal_ene);
    Scale_List(pm->d_reciprocal_ene, 0.5f, 1);

    Launch_Device_Kernel(
        charge_square_kernel,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers, charge,
        pm->charge_square);
    Sum_Of_List(pm->charge_square, pm->d_self_ene, pm->atom_numbers);
    Scale_List(pm->d_self_ene, -pm->esp.self_energy_coeff, 1);

    Launch_Device_Kernel(PME_Add_Energy_To_Potential, 1, 1, 0, NULL,
                         d_potential, pm->d_self_ene, pm->d_reciprocal_ene);
}

static void Run_PME_Reciprocal_Force_Backend(
    Particle_Mesh* pm, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, VECTOR* force, int need_virial,
    LTMatrix3* d_virial, int step)
{
    if (step % pm->update_interval != 0)
    {
        return;
    }

    deviceMemset(pm->PME_Q, 0, sizeof(float) * pm->PME_Nall);
    Launch_Device_Kernel(
        PME_Atom_Near,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, crd, pm->PME_atom_near,
        pm->PME_Nin, cell, rcell, pm->atom_numbers, pm->fftx, pm->ffty,
        pm->fftz, pm->PME_uxyz, pm->PME_frxyz, pm->force_backup);

    dim3 blockSize = {CONTROLLER::device_max_thread / 64, 64};
    Launch_Device_Kernel(PME_Q_Spread,
                         (pm->atom_numbers + blockSize.x - 1) / blockSize.x,
                         blockSize, 0, NULL, pm->PME_atom_near, charge,
                         pm->PME_frxyz, pm->PME_Q, pm->atom_numbers,
                         pm->PME_Nall);

    SPONGE_FFT_WRAPPER::R2C(pm->PME_plan_r2c, pm->PME_Q, pm->PME_FQ);

    blockSize = {CONTROLLER::device_warp,
                 CONTROLLER::device_max_thread / CONTROLLER::device_warp};
    if (need_virial)
    {
        Launch_Device_Kernel(
            PME_Sum_Virial,
            (pm->PME_Nfft + 4 * CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            blockSize, 0, NULL, pm->PME_Nfft, pm->PME_Virial_BC, pm->PME_FQ,
            d_virial, pm->fftz);
    }

    Launch_Device_Kernel(
        PME_BCFQ,
        (pm->PME_Nfft + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->PME_FQ, pm->PME_BC,
        pm->PME_Nfft);

    SPONGE_FFT_WRAPPER::C2R(pm->PME_plan_c2r, pm->PME_FQ, pm->PME_FBCFQ);

    blockSize = {8, CONTROLLER::device_max_thread / 8};
    Launch_Device_Kernel(PME_Final,
                         (pm->atom_numbers + blockSize.x - 1) / blockSize.x,
                         blockSize, 0, NULL, pm->PME_atom_near, charge,
                         pm->PME_FBCFQ, pm->force_backup, pm->PME_frxyz, rcell,
                         pm->fftx, pm->ffty, pm->fftz, pm->atom_numbers,
                         pm->PME_Nall);

    Launch_Device_Kernel(
        device_add_force,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers,
        pm->update_interval, force, pm->force_backup);
}

static void Run_PME_Reciprocal_Energy_Backend(Particle_Mesh* pm,
                                              const float* charge,
                                              float* d_potential)
{
    Launch_Device_Kernel(PME_Energy_Product, 1, CONTROLLER::device_max_thread,
                         0, NULL, pm->PME_Nall, pm->PME_Q, pm->PME_FBCFQ,
                         pm->d_reciprocal_ene);
    Scale_List(pm->d_reciprocal_ene, 0.5f, 1);

    Launch_Device_Kernel(
        charge_square_kernel,
        (pm->atom_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, NULL, pm->atom_numbers, charge,
        pm->charge_square);
    Sum_Of_List(pm->charge_square, pm->d_self_ene, pm->atom_numbers);
    Scale_List(pm->d_self_ene, -pm->beta / sqrt(PI), 1);

    Sum_Of_List(charge, pm->charge_sum, pm->atom_numbers);
    Launch_Device_Kernel(device_add, 1, 1, 0, NULL, pm->d_self_ene,
                         pm->neutralizing_factor, pm->charge_sum);
    Launch_Device_Kernel(PME_Add_Energy_To_Potential, 1, 1, 0, NULL,
                         d_potential, pm->d_self_ene, pm->d_reciprocal_ene);
}

static void Update_ESP_Box_Backend(Particle_Mesh* pm, LTMatrix3 rcell,
                                   float volume)
{
    dim3 blockSize = {8, 8, CONTROLLER::device_max_thread / 64};
    dim3 gridSize = {64, 64};
    pm->neutralizing_factor = 0.0f;
    Launch_Device_Kernel(
        up_box_esp_bc, gridSize, blockSize, 0, NULL, pm->fftx, pm->ffty,
        pm->fftz, pm->ESP_BC, pm->ESP_BC0, pm->ESP_Virial_BC, rcell, volume,
        pm->esp.cutoff, pm->esp.c_split, pm->esp.table_points,
        pm->esp.split_poly_order, pm->esp.table_mode == ESPTableMode::POLY,
        pm->ESP_split_fourier_table, pm->ESP_split_fourier_coeff,
        pm->ESP_split_fourier_derivative_table,
        pm->ESP_split_fourier_derivative_coeff);
}

static void Update_PME_Box_Backend(Particle_Mesh* pm, LTMatrix3 rcell,
                                   float volume)
{
    dim3 blockSize = {8, 8, CONTROLLER::device_max_thread / 64};
    dim3 gridSize = {64, 64};
    pm->neutralizing_factor = -0.5 * CONSTANT_Pi / (pm->beta * pm->beta * volume);
    float mprefactor = PI * PI / -pm->beta / pm->beta;
    Launch_Device_Kernel(up_box_bc, gridSize, blockSize, 0, NULL, pm->fftx,
                         pm->ffty, pm->fftz, pm->PME_BC, pm->PME_BC0,
                         pm->PME_Virial_BC, mprefactor, rcell, volume);
}

static void Scale_Box_Corners(Particle_Mesh* pm, LTMatrix3 g, float dt)
{
    Scale_Positions_Device(g, &pm->min_corner, dt);
    Scale_Positions_Device(g, &pm->max_corner, dt);
}

void Particle_Mesh::PME_Reciprocal_Force_With_Energy_And_Virial(
    const VECTOR* crd, const LTMatrix3 cell, const LTMatrix3 rcell,
    const float* charge, VECTOR* force, int need_virial, int need_energy,
    LTMatrix3* d_virial, float* d_potential, int step)
{
    if (is_initialized && calculate_reciprocal_part)
    {
        if (need_energy)
        {
            deviceMemset(d_reciprocal_ene, 0, sizeof(float));
            deviceMemset(d_self_ene, 0, sizeof(float));
        }
        if (backend == ParticleMeshBackend::ESP)
        {
            Run_ESP_Reciprocal_Force_Backend(this, crd, cell, rcell, charge,
                                             force, need_virial, d_virial,
                                             step);
            if (need_energy)
            {
                Run_ESP_Reciprocal_Energy_Backend(this, charge, d_potential);
            }
            return;
        }
        Run_PME_Reciprocal_Force_Backend(this, crd, cell, rcell, charge, force,
                                         need_virial, d_virial, step);
        if (need_energy)
        {
            Run_PME_Reciprocal_Energy_Backend(this, charge, d_potential);
        }
    }
}

// 计算PME的位移势能和Virial张量
static __global__ void up_box_bc(int fftx, int ffty, int fftz, float* PME_BC,
                                 float* PME_BC0, LTMatrix3* PME_virial_BC,
                                 float mprefactor, LTMatrix3 rcell,
                                 float volume)
{
    float kxrp, kyrp, kzrp;
    int ky, kz, index;
    float msq;
    VECTOR m;
    LTMatrix3 virial_bc_local;
    float bc_local;
#ifdef USE_GPU
    for (int kx = blockIdx.x * blockDim.x + threadIdx.x; kx < fftx;
         kx += blockDim.x * gridDim.x)
#else
#pragma omp parallel for firstprivate(kxrp, kyrp, kzrp, ky, kz, index, msq, m, \
                                          virial_bc_local, bc_local)
    for (int kx = 0; kx < fftx; kx++)
#endif
    {
        kxrp = kx;
        if (kx > fftx / 2) kxrp = kx - fftx;
#ifdef USE_GPU
        for (ky = blockIdx.y * blockDim.y + threadIdx.y; ky < ffty;
             ky += blockDim.y * gridDim.y)
#else
        for (ky = 0; ky < ffty; ky++)
#endif
        {
            kyrp = ky;
            if (ky > ffty / 2) kyrp = ky - ffty;
#ifdef USE_GPU
            for (kz = threadIdx.z; kz <= fftz / 2; kz += blockDim.z)
#else
            for (kz = 0; kz <= fftz / 2; kz++)
#endif
            {
                kzrp = kz;
                m = {kxrp, kyrp, kzrp};
                m = MultiplyTranspose(m, rcell);
                msq = m * m;

                index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

                if (kx + ky + kz == 0)
                {
                    PME_BC[index] = 0;
                    PME_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                }
                else
                {
                    bc_local = (float)1.0 / PI / msq * exp(mprefactor * msq) /
                               volume * PME_BC0[index];
                    virial_bc_local.a11 =
                        1 - 2 / msq * (1 + mprefactor * msq) * m.x * m.x;
                    virial_bc_local.a21 =
                        0 - 2 / msq * (1 + mprefactor * msq) * m.y * m.x;
                    virial_bc_local.a22 =
                        1 - 2 / msq * (1 + mprefactor * msq) * m.y * m.y;
                    virial_bc_local.a31 =
                        0 - 2 / msq * (1 + mprefactor * msq) * m.z * m.x;
                    virial_bc_local.a32 =
                        0 - 2 / msq * (1 + mprefactor * msq) * m.z * m.y;
                    virial_bc_local.a33 =
                        1 - 2 / msq * (1 + mprefactor * msq) * m.z * m.z;
                    PME_virial_BC[index] = 0.5f * bc_local * virial_bc_local;
                    PME_BC[index] = bc_local;
                }
            }
        }
    }
}

static __global__ void up_box_esp_bc(
    int fftx, int ffty, int fftz, float* ESP_BC, const float* ESP_BC0,
    LTMatrix3* ESP_virial_BC, LTMatrix3 rcell, float volume, float cutoff,
    float c_split, int table_points, int split_poly_order, int use_poly,
    const float* split_fourier_table, const float* split_fourier_coeff,
    const float* split_fourier_derivative_table,
    const float* split_fourier_derivative_coeff)
{
    float kxrp, kyrp, kzrp;
    int ky, kz, index;
    float msq;
    VECTOR m;
    LTMatrix3 virial_bc_local;
    float bc_local;
    float split_fourier, split_fourier_derivative;
    float derivative_ratio, metric_factor;
    const float split_scale = 2.0f * CONSTANT_Pi * cutoff / c_split;
#ifdef USE_GPU
    for (int kx = blockIdx.x * blockDim.x + threadIdx.x; kx < fftx;
         kx += blockDim.x * gridDim.x)
#else
#pragma omp parallel for firstprivate(                                      \
        kxrp, kyrp, kzrp, ky, kz, index, msq, m, virial_bc_local, bc_local, \
            split_fourier, split_fourier_derivative, derivative_ratio,      \
            metric_factor)
    for (int kx = 0; kx < fftx; kx++)
#endif
    {
        kxrp = kx;
        if (kx > fftx / 2) kxrp = kx - fftx;
#ifdef USE_GPU
        for (ky = blockIdx.y * blockDim.y + threadIdx.y; ky < ffty;
             ky += blockDim.y * gridDim.y)
#else
        for (ky = 0; ky < ffty; ky++)
#endif
        {
            kyrp = ky;
            if (ky > ffty / 2) kyrp = ky - ffty;
#ifdef USE_GPU
            for (kz = threadIdx.z; kz <= fftz / 2; kz += blockDim.z)
#else
            for (kz = 0; kz <= fftz / 2; kz++)
#endif
            {
                kzrp = kz;
                m = {kxrp, kyrp, kzrp};
                m = MultiplyTranspose(m, rcell);
                msq = m * m;

                index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;

                ESP_BC[index] = 0.0f;
                ESP_virial_BC[index] = {0, 0, 0, 0, 0, 0};
                if (kx + ky + kz == 0 || msq <= 0.0f)
                {
                    continue;
                }

                float split_arg = split_scale * sqrtf(msq);
                if (split_arg > 1.0f)
                {
                    continue;
                }

                split_fourier =
                    0.5f * ESP_Eval_Direct_Scalar(
                               split_fourier_table, split_fourier_coeff,
                               table_points, split_poly_order, use_poly,
                               split_arg);
                split_fourier_derivative =
                    0.5f * ESP_Eval_Direct_Scalar(
                               split_fourier_derivative_table,
                               split_fourier_derivative_coeff, table_points,
                               split_poly_order, use_poly, split_arg);
                if (!ESP_Float_Is_Bounded(ESP_BC0[index], 1.0e12f))
                {
                    continue;
                }
                bc_local = split_fourier * ESP_BC0[index] /
                           (CONSTANT_Pi * msq * volume);
                if (!ESP_Float_Is_Bounded(bc_local, 1.0e6f))
                {
                    ESP_BC[index] = 0.0f;
                    continue;
                }
                ESP_BC[index] = bc_local;

                derivative_ratio = 0.0f;
                if (fabsf(split_fourier) > 1.0e-30f)
                {
                    derivative_ratio = split_fourier_derivative / split_fourier;
                }
                metric_factor = (2.0f - derivative_ratio) / msq;
                virial_bc_local.a11 = 1.0f - metric_factor * m.x * m.x;
                virial_bc_local.a21 = 0.0f - metric_factor * m.y * m.x;
                virial_bc_local.a22 = 1.0f - metric_factor * m.y * m.y;
                virial_bc_local.a31 = 0.0f - metric_factor * m.z * m.x;
                virial_bc_local.a32 = 0.0f - metric_factor * m.z * m.y;
                virial_bc_local.a33 = 1.0f - metric_factor * m.z * m.z;
                ESP_virial_BC[index] = 0.5f * bc_local * virial_bc_local;
            }
        }
    }
}

static void Scale_Positions_Device(const LTMatrix3 g, VECTOR* crd, float dt)
{
    VECTOR r_dash;
    r_dash.x = crd[0].x +
               dt * (crd[0].x * g.a11 + crd[0].y * g.a21 + crd[0].z * g.a31);
    r_dash.y = crd[0].y + dt * (crd[0].y * g.a22 + crd[0].z * g.a32);
    r_dash.z = crd[0].z + dt * crd[0].z * g.a33;
    crd[0] = r_dash;
}

void Particle_Mesh::Update_Box(LTMatrix3 cell, LTMatrix3 rcell, LTMatrix3 g,
                               float dt)
{
    float volume = cell.a11 * cell.a22 * cell.a33;
    if (backend == ParticleMeshBackend::ESP)
    {
        Update_ESP_Box_Backend(this, rcell, volume);
        Scale_Box_Corners(this, g, dt);
        return;
    }
    Update_PME_Box_Backend(this, rcell, volume);
    Scale_Box_Corners(this, g, dt);
}

//-------domain-decomposition and communication----------------

// 找出n的所有因子，存入factor_set，并按从大到小排序
static void find_factor(std::vector<int>& factor_set, int n)
{
    for (int i = 1; i <= std::sqrt(n); ++i)
    {
        if (n % i == 0)
        {
            factor_set.push_back(i);
            if (i != n / i)
            {
                factor_set.push_back(n / i);
            }
        }
    }
    std::sort(factor_set.begin(), factor_set.end(), std::greater<int>());
}

// Domain Decomposition
void Particle_Mesh::Domain_Decomposition(CONTROLLER* controller,
                                         VECTOR box_length,
                                         INT_VECTOR pp_split_num)
{
    // 如果设置PM进程数=0， 直接返回；区域分割只在主进程上做一次
    if (controller->MPI_rank != 0 || !PM_MPI_size)
    {
        return;
    }
    // 如果PP进程数不能被PM进程数整除，报错
    if (controller->PP_MPI_size % PM_MPI_size)
    {
        controller->Throw_SPONGE_Error(
            spongeErrorValueErrorCommand,
            "Particle_Mesh_Ewald::Domain_Decomposition",
            "Reason:\n\tThe number of PP processes must be divisible by the "
            "number of PM processes.");
        return;
    }

    int nx = pp_split_num.int_x;
    int ny = pp_split_num.int_y;
    int nz = pp_split_num.int_z;

    std::vector<int> fac_set_x;
    std::vector<int> fac_set_y;
    std::vector<int> fac_set_z;
    find_factor(fac_set_x, nx);
    find_factor(fac_set_y, ny);
    find_factor(fac_set_z, nz);
    int pm_size;
    for (int& tmpx : fac_set_x)
    {
        pm_size = controller->PM_MPI_size;
        if (pm_size % tmpx == 0 && pm_dom_dec_split_num.int_x == 0)
        {
            pm_size /= tmpx;
            for (int& tmpy : fac_set_y)
            {
                if (pm_size % tmpy == 0 && pm_dom_dec_split_num.int_y == 0)
                {
                    pm_size /= tmpy;
                    for (int& tmpz : fac_set_z)
                    {
                        if (pm_size == tmpz)
                        {
                            pm_dom_dec_split_num.int_x = tmpx;
                            pm_dom_dec_split_num.int_y = tmpy;
                            pm_dom_dec_split_num.int_z = tmpz;
                        }
                    }
                }
            }
        }
    }

    nx = pm_dom_dec_split_num.int_x;
    ny = pm_dom_dec_split_num.int_y;
    nz = pm_dom_dec_split_num.int_z;
    std::cout << "pm_nx= " << nx << ", pm_ny= " << ny << ", pm_nz= " << nz
              << std::endl;
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int rank_id = i + j * nx + k * nx * ny;
                min_corner_set[rank_id].x = box_length.x / nx * i;
                min_corner_set[rank_id].y = box_length.y / ny * j;
                min_corner_set[rank_id].z = box_length.z / nz * k;
                max_corner_set[rank_id].x = box_length.x / nx * (i + 1);
                max_corner_set[rank_id].y = box_length.y / ny * (j + 1);
                max_corner_set[rank_id].z = box_length.z / nz * (k + 1);
            }
        }
    }

    // 若单进程， PM与PP共享同一进程下，
    if (controller->MPI_size == 1 && PM_MPI_size == 1)
    {
        pm_pp_num[0] = 1;
        pm_pp_corres[0][0] = 0;
        return;
    }

    int nx_ = pp_split_num.int_x / pm_dom_dec_split_num.int_x;
    int ny_ = pp_split_num.int_y / pm_dom_dec_split_num.int_y;
    int nz_ = pp_split_num.int_z / pm_dom_dec_split_num.int_z;
    for (int i = 0; i < controller->PM_MPI_size; ++i)
    {
        pm_pp_num[i] = 0;
    }
    for (int k = 0; k < pp_split_num.int_z; ++k)
    {
        for (int j = 0; j < pp_split_num.int_y; ++j)
        {
            for (int i = 0; i < pp_split_num.int_x; ++i)
            {
                int pp_rank_id = i + j * pp_split_num.int_x +
                                 k * pp_split_num.int_x * pp_split_num.int_y;
                int pm_rank_id = i / nx_ +
                                 j / ny_ * pm_dom_dec_split_num.int_x +
                                 k / nz_ * pm_dom_dec_split_num.int_x *
                                     pm_dom_dec_split_num.int_y;
                pm_pp_corres[pm_rank_id][pm_pp_num[pm_rank_id]] = pp_rank_id;
                pm_pp_num[pm_rank_id]++;
            }
        }
    }
}

void Particle_Mesh::Send_Recv_Dom_Dec(CONTROLLER* controller)
{
    // 如果设置PM进程数=0， 直接返回
    if (!PM_MPI_size)
    {
        return;
    }
    // 如果PM与PP共用一个进程，则不需要通信
    if (controller->MPI_size == 1 && PM_MPI_size == 1)
    {
        strcpy(this->FFT_MPI_TYPE, "DISABLE");
        return;
    }
#ifdef USE_MPI
    // PP进程与PM进程分割
    if (controller->MPI_rank == 0)
    {
        // 发送PP进程对应的PM进程号
        for (int pm_id = 0; pm_id < controller->PM_MPI_size; ++pm_id)
        {
            int pm_rank_tot = pm_id + controller->PP_MPI_size;
            for (int i = 0; i < pm_pp_num[pm_id]; ++i)
            {
                if (pm_pp_corres[pm_id][i] != 0)
                {
                    MPI_Send(&pm_rank_tot, sizeof(int), MPI_BYTE,
                             pm_pp_corres[pm_id][i], pm_pp_corres[pm_id][i],
                             MPI_COMM_WORLD);
                }
                else
                {
                    pp_corres_pm_rank = pm_rank_tot;
                }
            }
        }
        // 发送PM进程对应的pp进程数与进程索引集合；发送域分割信息
        for (int pm_id = 0; pm_id < controller->PM_MPI_size; ++pm_id)
        {
            int pm_rank_tot = pm_id + controller->PP_MPI_size;
            MPI_Send(&min_corner_set[pm_id], sizeof(VECTOR), MPI_BYTE,
                     pm_rank_tot, 0, MPI_COMM_WORLD);
            MPI_Send(&max_corner_set[pm_id], sizeof(VECTOR), MPI_BYTE,
                     pm_rank_tot, 1, MPI_COMM_WORLD);
            MPI_Send(&pm_pp_num[pm_id], sizeof(int), MPI_BYTE, pm_rank_tot, 2,
                     MPI_COMM_WORLD);
            MPI_Send(pm_pp_corres[pm_id], pm_pp_num[pm_id] * sizeof(int),
                     MPI_BYTE, pm_rank_tot, 3, MPI_COMM_WORLD);
            MPI_Send(&pm_dom_dec_split_num, sizeof(INT_VECTOR), MPI_BYTE,
                     pm_rank_tot, 4, MPI_COMM_WORLD);
        }
    }
    else
    {
        if (controller->MPI_rank < controller->PP_MPI_size)
        {
            MPI_Recv(&pp_corres_pm_rank, sizeof(int), MPI_BYTE, 0,
                     controller->MPI_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            MPI_Recv(&min_corner, sizeof(VECTOR), MPI_BYTE, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&max_corner, sizeof(VECTOR), MPI_BYTE, 0, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&pm_corres_pp_num, sizeof(int), MPI_BYTE, 0, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pm_corres_pp_rank_set, pm_corres_pp_num * sizeof(int),
                     MPI_BYTE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&pm_dom_dec_split_num, sizeof(INT_VECTOR), MPI_BYTE, 0, 4,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (pm_dom_dec_split_num.int_y == 1 &&
                pm_dom_dec_split_num.int_z == 1)
            {
                strcpy(this->FFT_MPI_TYPE, "SLAB");
            }
            else
            {
                strcpy(this->FFT_MPI_TYPE, "BRICK");
            }
        }
    }
#endif
}

void Particle_Mesh::Find_Neighbor_Domain(CONTROLLER* controller)
{
    if (!PM_MPI_size)
    {
        return;
    }
    if (controller->PM_MPI_size == 1 ||
        controller->MPI_rank < controller->PP_MPI_size)
    {
        for (int dir = 0; dir < 6; ++dir)
        {
            neighbor_num[dir] = 0;
        }
        return;
    }
    int nx = pm_dom_dec_split_num.int_x;
    int ny = pm_dom_dec_split_num.int_y;
    int nz = pm_dom_dec_split_num.int_z;

    neighbor_num[0] = nx == 1 ? 0 : 1;
    neighbor_num[1] = nx == 1 ? 0 : 1;
    neighbor_num[2] = ny == 1 ? 0 : 1;
    neighbor_num[3] = ny == 1 ? 0 : 1;
    neighbor_num[4] = nz == 1 ? 0 : 1;
    neighbor_num[5] = nz == 1 ? 0 : 1;

    int rank_id = pm_rank;
    int i = rank_id % (nx);
    int j = (rank_id / nx) % ny;
    int k = rank_id / (nx * ny);

    if (nx > 1)
    {
        neighbor_dir[0][0] = (i + 1) % nx + j * nx + k * nx * ny;
        neighbor_dir[1][0] = (i - 1 + nx) % nx + j * nx + k * nx * ny;
    }

    if (ny > 1)
    {
        neighbor_dir[2][0] = i % nx + ((j + 1) % ny) * nx + k * nx * ny;
        neighbor_dir[3][0] = i % nx + ((j - 1 + ny) % ny) * nx + k * nx * ny;
    }

    if (nz > 1)
    {
        neighbor_dir[4][0] = i % nx + j * nx + ((k + 1) % nz) * nx * ny;
        neighbor_dir[5][0] = i % nx + j * nx + ((k - 1 + nz) % nz) * nx * ny;
    }
}

static __global__ void inverse_global_and_local(const int* A, int* B, int N)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
#else
#pragma omp parallel for
    for (int i = 0; i < N; i++)
#endif
    {
        int v = A[i];  // v ∈ [0, N)
        B[v] = i;
    }
}

static __global__ void crd_local_to_global(VECTOR* l_crd, VECTOR* g_crd,
                                           int* atom_id_l_g, int N)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
#else
#pragma omp parallel for
    for (int i = 0; i < N; i++)
#endif
    {
        int g_id = atom_id_l_g[i];
        g_crd[g_id] = l_crd[i];
    }
}

void Particle_Mesh::Get_Atoms(CONTROLLER* controller, VECTOR* pme_crd,
                              float* pme_charge, int pp_atom_numbers,
                              VECTOR* pp_crd, float* pp_charge, int* atom_local,
                              bool atom_number_label, bool charge_label,
                              bool crd_label, bool id_label)
{
    if (!PM_MPI_size)
    {
        return;
    }
    // 若单进程PM与PP共享同一进程下，什么也不做。共享进程下直接传入dd.crd等计算，不再重复拷贝内存
    if (controller->MPI_size == 1 && PM_MPI_size == 1)
    {
        return;
    }
#ifdef USE_MPI
    // 先阻塞通信原子数
    if (controller->MPI_rank < controller->PP_MPI_size)
    {
        if (atom_number_label)
        {
            MPI_Send(&pp_atom_numbers, sizeof(int), MPI_BYTE, pp_corres_pm_rank,
                     0, MPI_COMM_WORLD);
        }
    }
    else
    {
        if (atom_number_label)
        {
            int prefix = 0;
            // 接收PP进程对应的原子数
            for (int i = 0; i < pm_corres_pp_num; ++i)
            {
                int pp_rank = pm_corres_pp_rank_set[i];
                {
                    MPI_Recv(&pm_corres_pp_atom_number[i], sizeof(int),
                             MPI_BYTE, pp_rank, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    pm_corres_pp_atom_number_prefix[i] = prefix;
                    prefix += pm_corres_pp_atom_number[i];
                }
                this->atom_numbers = prefix;
            }
        }
    }
    // 通信坐标与电荷

    if (controller->MPI_rank < controller->PP_MPI_size)
    {
        D_MPI_GroupStart();
        if (crd_label)
        {
            D_MPI_Send(pp_crd, pp_atom_numbers * sizeof(VECTOR), D_MPI_BYTE,
                       pp_corres_pm_rank, 1, controller->D_MPI_COMM_WORLD,
                       pm_stream);
        }
        if (charge_label)
        {
            D_MPI_Send(pp_charge, pp_atom_numbers * sizeof(float), D_MPI_BYTE,
                       pp_corres_pm_rank, 2, controller->D_MPI_COMM_WORLD,
                       pm_stream);
        }
        if (id_label)
        {
            D_MPI_Send(atom_local, pp_atom_numbers * sizeof(int), D_MPI_BYTE,
                       pp_corres_pm_rank, 3, controller->D_MPI_COMM_WORLD,
                       pm_stream);
        }
        D_MPI_GroupEnd();
#ifdef USE_GPU
        deviceStreamSynchronize(pm_stream);
#endif
    }
    else
    {
        D_MPI_GroupStart();
        for (int i = 0; i < pm_corres_pp_num; ++i)
        {
            int pp_rank = pm_corres_pp_rank_set[i];
            if (crd_label)
            {
                D_MPI_Recv(pme_crd + pm_corres_pp_atom_number_prefix[i],
                           pm_corres_pp_atom_number[i] * sizeof(VECTOR),
                           D_MPI_BYTE, pp_rank, 1, controller->D_MPI_COMM_WORLD,
                           pm_stream);
            }
            if (charge_label)
            {
                D_MPI_Recv(pme_charge + pm_corres_pp_atom_number_prefix[i],
                           pm_corres_pp_atom_number[i] * sizeof(float),
                           D_MPI_BYTE, pp_rank, 2, controller->D_MPI_COMM_WORLD,
                           pm_stream);
            }
            if (id_label)
            {
                D_MPI_Recv(atom_id_l_g + pm_corres_pp_atom_number_prefix[i],
                           pm_corres_pp_atom_number[i] * sizeof(int),
                           D_MPI_BYTE, pp_rank, 3, controller->D_MPI_COMM_WORLD,
                           pm_stream);
            }
        }
        D_MPI_GroupEnd();
#ifdef USE_GPU
        deviceStreamSynchronize(pm_stream);
#endif
        // 反转local与global的映射关系
        if (id_label)
        {
            Launch_Device_Kernel(
                inverse_global_and_local,
                (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, atom_id_l_g,
                atom_id_g_l, this->atom_numbers);
        }
        if (crd_label)
        {
            Launch_Device_Kernel(
                crd_local_to_global,
                (this->atom_numbers + CONTROLLER::device_max_thread - 1) /
                    CONTROLLER::device_max_thread,
                CONTROLLER::device_max_thread, 0, NULL, pme_crd, g_crd,
                atom_id_l_g, this->atom_numbers);
        }
    }
#endif
}

// 目前只做单进程PME，暂时不考虑ghost，同样也不考虑get_local

void Particle_Mesh::Send_Recv_Force(CONTROLLER* controller, VECTOR* frc,
                                    VECTOR* pp_frc, int pp_atom_numbers)
{
    if (!PM_MPI_size)
    {
        return;
    }
#ifdef USE_MPI

    if (controller->MPI_rank < controller->PP_MPI_size)
    {
        D_MPI_GroupStart();
        D_MPI_Recv(frc, sizeof(VECTOR) * pp_atom_numbers, D_MPI_BYTE,
                   pp_corres_pm_rank, controller->MPI_rank,
                   controller->D_MPI_COMM_WORLD, pm_stream);
        D_MPI_GroupEnd();
#ifdef USE_GPU
        deviceStreamSynchronize(pm_stream);
#endif
        Launch_Device_Kernel(
            device_add_force,
            (pp_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, pp_atom_numbers, 1, pp_frc,
            frc);
    }
    else
    {
        for (int i = 0; i < pm_corres_pp_num; ++i)
        {
            int pp_rank = pm_corres_pp_rank_set[i];
            int prefix = pm_corres_pp_atom_number_prefix[i];
            D_MPI_GroupStart();
            D_MPI_Send(frc + prefix,
                       sizeof(VECTOR) * pm_corres_pp_atom_number[i], D_MPI_BYTE,
                       pp_rank, pp_rank, controller->D_MPI_COMM_WORLD,
                       pm_stream);
            D_MPI_GroupEnd();
#ifdef USE_GPU
            deviceStreamSynchronize(pm_stream);
#endif
        }
    }
#endif
}

void Particle_Mesh::Create_Stream() { deviceStreamCreate(&pm_stream); }

void Particle_Mesh::Destroy_Stream() { deviceStreamDestroy(pm_stream); }

static __global__ void MPI_PME_Excluded_Force_With_Atom_Energy_Correction(
    const int atom_numbers, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const float pme_beta,
    const int* excluded_list_start, const int* excluded_list,
    const int* excluded_atom_numbers, VECTOR* frc, float* atom_ene,
    float* this_ene, LTMatrix3* atom_virial, int need_energy, int need_virial,
    const int* local2global, const int* global2local, const float factor)
{
    SIMPLE_DEVICE_FOR(local_i, atom_numbers)
    {
        int global_i = local_i;
        int excluded_numbers = excluded_atom_numbers[global_i];
        if (excluded_numbers > 0)
        {
            int list_start = excluded_list_start[global_i];
            int list_end = list_start + excluded_numbers;
            int local_j, global_j;

            float charge_i = charge[local_i];
            float charge_j;
            float dr_abs;
            float beta_dr;

            VECTOR r1 = crd[local_i], r2;
            VECTOR dr;
            float dr2;

            float frc_abs = 0.;
            VECTOR frc_lin;
            VECTOR frc_record = {0., 0., 0.};
            LTMatrix3 virial_record = {0, 0, 0, 0, 0, 0};
            float ene_lin = 0.;

            for (int i = list_start; i < list_end; i = i + 1)
            {
                global_j = excluded_list[i];
                local_j = global_j;
                r2 = crd[local_j];
                charge_j = charge[local_j];

                dr = Get_Periodic_Displacement(r2, r1, cell, rcell);
                dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                // 假设剔除表中的原子对距离总是小于cutoff的，正常体系

                dr_abs = sqrtf(dr2);
                beta_dr = pme_beta * dr_abs;
                frc_abs = beta_dr * TWO_DIVIDED_BY_SQRT_PI *
                              expf(-beta_dr * beta_dr) +
                          erfcf(beta_dr);
                frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
                frc_abs = -charge_i * charge_j * frc_abs;
                frc_lin = frc_abs * dr;
                if (factor > 0.6f) atomicAdd(frc + local_j, -frc_lin);
                frc_record = frc_record + frc_lin;
                if (need_energy)
                    ene_lin -=
                        factor * charge_i * charge_j * erff(beta_dr) / dr_abs;
                if (need_virial)
                    virial_record =
                        virial_record -
                        factor * Get_Virial_From_Force_Dis(frc_lin, dr);
            }  // atom_j cycle
            atomicAdd(frc + local_i, frc_record);
            if (need_energy)
            {
                atomicAdd(atom_ene + local_i, ene_lin);
                this_ene[local_i] = ene_lin;
            }
            if (need_virial) atomicAdd(atom_virial + local_i, virial_record);
        }  // if need excluded
    }
}

void Particle_Mesh::MPI_PME_Excluded_Force_With_Atom_Energy(
    const int local_atom_numbers, const int* atom_local,
    const int* atom_local_id, const VECTOR* crd, const LTMatrix3 cell,
    const LTMatrix3 rcell, const float* charge, const int* excluded_list_start,
    const int* excluded_list, const int* excluded_atom_numbers, VECTOR* frc,
    int need_energy, float* atom_ene, int need_virial, LTMatrix3* atom_virial)
{
    if (is_initialized && calculate_excluded_part)
    {
        if (need_energy)
            deviceMemset(d_correction_atom_energy, 0,
                         sizeof(float) * local_atom_numbers);

        Launch_Device_Kernel(
            MPI_PME_Excluded_Force_With_Atom_Energy_Correction,
            (local_atom_numbers + CONTROLLER::device_max_thread - 1) /
                CONTROLLER::device_max_thread,
            CONTROLLER::device_max_thread, 0, NULL, local_atom_numbers, crd,
            cell, rcell, charge, beta, excluded_list_start, excluded_list,
            excluded_atom_numbers, frc, atom_ene, d_correction_atom_energy,
            atom_virial, need_energy, need_virial, atom_local, atom_local_id,
            exclude_factor);
    }
}

void Particle_Mesh::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized) return;
    // 单进程, PM与PP共享同一进程情况
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        Sum_Of_List(d_correction_atom_energy, d_correction_ene, atom_numbers);
        Sum_Of_List(d_direct_atom_energy, d_direct_ene, atom_numbers);
        deviceMemcpy(&direct_ene, d_direct_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&correction_ene, d_correction_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&self_ene, d_self_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        ee_ene = direct_ene + reciprocal_ene + self_ene + correction_ene;
        controller->Step_Print("PM", ee_ene, true);
        if (print_detail)
        {
            controller->Step_Print("PM_direct", direct_ene);
            controller->Step_Print("PM_reciprocal", reciprocal_ene);
            controller->Step_Print("PM_self", self_ene);
            controller->Step_Print("PM_correction", correction_ene);
        }
        return;
    }
    if (CONTROLLER::MPI_rank < CONTROLLER::PP_MPI_size)
    {
        Sum_Of_List(d_correction_atom_energy, d_correction_ene, atom_numbers);
        Sum_Of_List(d_direct_atom_energy, d_direct_ene, atom_numbers);
        self_ene = 0;
        reciprocal_ene = 0;
        deviceMemcpy(&direct_ene, d_direct_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&correction_ene, d_correction_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    else
    {
        direct_ene = 0;
        correction_ene = 0;
        deviceMemcpy(&self_ene, d_self_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
        deviceMemcpy(&reciprocal_ene, d_reciprocal_ene, sizeof(float),
                     deviceMemcpyDeviceToHost);
    }
    ee_ene = direct_ene + reciprocal_ene + self_ene + correction_ene;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &ee_ene, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif
    controller->Step_Print("PM", ee_ene, true);
    if (print_detail)
    {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &self_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &reciprocal_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &direct_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &correction_ene, 1, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
#endif
        controller->Step_Print("PM_direct", direct_ene);
        controller->Step_Print("PM_reciprocal", reciprocal_ene);
        controller->Step_Print("PM_self", self_ene);
        controller->Step_Print("PM_correction", correction_ene);
    }
}

void Particle_Mesh::reset_global_force(
    int no_direct_interaction_virtual_atom_numbers)
{
    deviceMemset(g_frc, 0,
                 sizeof(VECTOR) * (atom_numbers +
                                   no_direct_interaction_virtual_atom_numbers));
}

static __global__ void add_global_to_local_force(const VECTOR* g_frc,
                                                 VECTOR* l_frc,
                                                 const int* atom_id_g_l, int N)
{
#ifdef USE_GPU
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
#else
#pragma omp parallel for
    for (int i = 0; i < N; i++)
#endif
    {
        int l_id = atom_id_g_l[i];
        l_frc[l_id] = l_frc[l_id] + g_frc[i];
    }
}

void Particle_Mesh::add_force_g_to_l(VECTOR* l_frc)
{
    Launch_Device_Kernel(add_global_to_local_force,
                         (atom_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, NULL, g_frc, l_frc,
                         atom_id_g_l, atom_numbers);
}
