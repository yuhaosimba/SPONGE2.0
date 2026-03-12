#include "FGM_Double_Layer.h"

#include <cstdlib>

namespace
{
constexpr const char* kDefaultModuleName = "FGM_Double_Layer";
constexpr float kPlateClamp = 1e-4f;
constexpr float kDistanceClamp2 = 1e-8f;

bool FGM_Block_Exists(CONTROLLER* controller, const char* module_name)
{
    static const char* keys[] = {"enable",
                                 "ion_serial_start",
                                 "ion_numbers",
                                 "z1",
                                 "z2",
                                 "ep1",
                                 "ep2",
                                 "Nx",
                                 "Ny",
                                 "Nz",
                                 "FFT",
                                 "green_force_refresh_interval",
                                 "sphere_pos_file_name"};
    for (const char* key : keys)
    {
        if (controller->Command_Exist(module_name, key))
        {
            return true;
        }
    }
    return false;
}

int Count_Sphere_Sampling_Positions(CONTROLLER* controller,
                                    const char* module_name,
                                    const char* file_name)
{
    FILE* file = fopen(file_name, "rb");
    if (file == nullptr)
    {
        char error_string[CHAR_LENGTH_MAX];
        snprintf(error_string, sizeof(error_string),
                 "Reason:\n\t%s requires sphere_pos_file_name, but \"%s\" "
                 "can not be opened.\n",
                 module_name, file_name);
        controller->Throw_SPONGE_Error(spongeErrorOpenFileFailed, module_name,
                                       error_string);
    }

    int point_numbers = 0;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    while (fscanf(file, "%f%f%f", &x, &y, &z) == 3)
    {
        point_numbers += 1;
    }
    if (point_numbers <= 0)
    {
        if (fseek(file, 0, SEEK_END) == 0)
        {
            const long file_size = ftell(file);
            if (file_size > 0 &&
                file_size % static_cast<long>(sizeof(float) * 3) == 0)
            {
                point_numbers =
                    static_cast<int>(file_size / (sizeof(float) * 3));
            }
        }
    }
    fclose(file);

    if (point_numbers <= 0)
    {
        char error_string[CHAR_LENGTH_MAX];
        snprintf(error_string, sizeof(error_string),
                 "Reason:\n\t%s sampling file \"%s\" does not contain any "
                 "valid points.\n",
                 module_name, file_name);
        controller->Throw_SPONGE_Error(spongeErrorBadFileFormat, module_name,
                                       error_string);
    }
    return point_numbers;
}

__host__ __device__ __forceinline__ float Wrap_Periodic_Component(float dr,
                                                                  float box)
{
    return dr - floorf(dr / box + 0.5f) * box;
}

__host__ __device__ __forceinline__ int Alternating_Sign(int n)
{
    return (abs(n) % 2 == 0) ? 1 : -1;
}

static __global__ void FGM_Reset_Selected_Atom_Energy(const int ion_numbers,
                                                      const int ion_start,
                                                      float* atom_energy)
{
    SIMPLE_DEVICE_FOR(i, ion_numbers) { atom_energy[ion_start + i] = 0.0f; }
}

static __global__ void FGM_Image_Green_Force_And_Energy(
    const int ion_numbers, const int ion_serial_start, const int ion_serial_end,
    const VECTOR* crd, const float* charge, const VECTOR box_length,
    const float z1, const float plate_distance, const float ep1,
    const float external_field_z, const int image_layer_numbers, VECTOR* frc,
    float* atom_energy, float* fgm_atom_energy)
{
    SIMPLE_DEVICE_FOR(ion_index, ion_numbers)
    {
        const int atom_i = ion_serial_start + ion_index;
        const float charge_i = charge[atom_i];
        const VECTOR crd_i = crd[atom_i];

        float zi = crd_i.z - z1;
        zi = fminf(fmaxf(zi, kPlateClamp), plate_distance - kPlateClamp);

        VECTOR force_i = {0.0f, 0.0f, 0.0f};
        float energy_i = charge_i * (ep1 - external_field_z * zi);
        force_i.z += charge_i * external_field_z;

        for (int atom_j = ion_serial_start; atom_j < ion_serial_end; atom_j++)
        {
            const float charge_j = charge[atom_j];
            if (charge_j == 0.0f)
            {
                continue;
            }

            const VECTOR crd_j = crd[atom_j];
            const float zj_raw = crd_j.z - z1;
            const float zj =
                fminf(fmaxf(zj_raw, kPlateClamp), plate_distance - kPlateClamp);

            const float dx =
                Wrap_Periodic_Component(crd_j.x - crd_i.x, box_length.x);
            const float dy =
                Wrap_Periodic_Component(crd_j.y - crd_i.y, box_length.y);

            for (int layer = -image_layer_numbers; layer <= image_layer_numbers;
                 layer++)
            {
                const float sign = static_cast<float>(Alternating_Sign(layer));

                const float z_plus = 2.0f * layer * plate_distance + zj;
                if (!(atom_j == atom_i && layer == 0))
                {
                    const float dz = z_plus - zi;
                    const float r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 > kDistanceClamp2)
                    {
                        const float inv_r = 1.0f / sqrtf(r2);
                        const float pair_charge = charge_i * sign * charge_j;
                        const float inv_r3 = inv_r * inv_r * inv_r;
                        force_i.x += pair_charge * dx * inv_r3;
                        force_i.y += pair_charge * dy * inv_r3;
                        force_i.z += pair_charge * dz * inv_r3;
                        energy_i += 0.5f * pair_charge * inv_r;
                    }
                }

                const float z_minus = 2.0f * layer * plate_distance - zj;
                const float dz_image = z_minus - zi;
                const float r2_image =
                    dx * dx + dy * dy + dz_image * dz_image;
                if (r2_image > kDistanceClamp2)
                {
                    const float inv_r = 1.0f / sqrtf(r2_image);
                    const float pair_charge = -charge_i * sign * charge_j;
                    const float inv_r3 = inv_r * inv_r * inv_r;
                    force_i.x += pair_charge * dx * inv_r3;
                    force_i.y += pair_charge * dy * inv_r3;
                    force_i.z += pair_charge * dz_image * inv_r3;
                    energy_i += 0.5f * pair_charge * inv_r;
                }
            }
        }

        atomicAdd(&frc[atom_i].x, force_i.x);
        atomicAdd(&frc[atom_i].y, force_i.y);
        atomicAdd(&frc[atom_i].z, force_i.z);
        atomicAdd(atom_energy + atom_i, energy_i);
        fgm_atom_energy[atom_i] = energy_i;
    }
}
}  // namespace

bool FGM_DOUBLE_LAYER::Is_Enabled_In_Controller(CONTROLLER* controller,
                                                const char* module_name)
{
    const char* name =
        module_name == nullptr ? kDefaultModuleName : module_name;
    if (!FGM_Block_Exists(controller, name))
    {
        return false;
    }
    if (!controller->Command_Exist(name, "enable"))
    {
        return true;
    }
    return controller->Get_Bool(name, "enable",
                                "FGM_DOUBLE_LAYER::Is_Enabled_In_Controller");
}

void FGM_DOUBLE_LAYER::Initial(CONTROLLER* controller, int atom_numbers_,
                               LTMatrix3 cell_, LTMatrix3 rcell_,
                               VECTOR box_length_, float cutoff_,
                               const char* module_name_)
{
    this->controller = controller;
    const char* final_module_name =
        module_name_ == nullptr ? kDefaultModuleName : module_name_;
    strcpy(module_name, final_module_name);

    if (!Is_Enabled_In_Controller(controller, module_name))
    {
        return;
    }

    atom_numbers = atom_numbers_;
    cell = cell_;
    rcell = rcell_;
    box_length = box_length_;
    cutoff = cutoff_;

    controller->printf("START INITIALIZING %s:\n", module_name);

    if (controller->Command_Exist(module_name, "ion_serial_start"))
    {
        controller->Check_Int(module_name, "ion_serial_start",
                              "FGM_DOUBLE_LAYER::Initial");
        ion_serial_start = atoi(controller->Command(module_name,
                                                    "ion_serial_start"));
    }
    if (controller->Command_Exist(module_name, "ion_numbers"))
    {
        controller->Check_Int(module_name, "ion_numbers",
                              "FGM_DOUBLE_LAYER::Initial");
        ion_numbers = atoi(controller->Command(module_name, "ion_numbers"));
    }
    else
    {
        ion_numbers = atom_numbers - ion_serial_start;
    }
    ion_serial_end = ion_serial_start + ion_numbers;
    if (ion_serial_start < 0 || ion_serial_start >= atom_numbers ||
        ion_numbers <= 0 || ion_serial_end > atom_numbers)
    {
        char error_string[CHAR_LENGTH_MAX];
        snprintf(error_string, sizeof(error_string),
                 "Reason:\n\t%s ion selection is out of range. "
                 "ion_serial_start=%d, ion_numbers=%d, atom_numbers=%d.\n",
                 module_name, ion_serial_start, ion_numbers, atom_numbers);
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "FGM_DOUBLE_LAYER::Initial",
                                       error_string);
    }

    if (controller->Command_Exist(module_name, "z1"))
    {
        controller->Check_Float(module_name, "z1", "FGM_DOUBLE_LAYER::Initial");
        z1 = atof(controller->Command(module_name, "z1"));
    }
    if (controller->Command_Exist(module_name, "z2"))
    {
        controller->Check_Float(module_name, "z2", "FGM_DOUBLE_LAYER::Initial");
        z2 = atof(controller->Command(module_name, "z2"));
    }
    else
    {
        z2 = box_length.z;
    }
    if (controller->Command_Exist(module_name, "ep1"))
    {
        controller->Check_Float(module_name, "ep1", "FGM_DOUBLE_LAYER::Initial");
        ep1 = atof(controller->Command(module_name, "ep1"));
    }
    if (controller->Command_Exist(module_name, "ep2"))
    {
        controller->Check_Float(module_name, "ep2", "FGM_DOUBLE_LAYER::Initial");
        ep2 = atof(controller->Command(module_name, "ep2"));
    }
    plate_distance = z2 - z1;
    if (plate_distance <= 2.0f * kPlateClamp)
    {
        char error_string[CHAR_LENGTH_MAX];
        snprintf(error_string, sizeof(error_string),
                 "Reason:\n\t%s requires z2 > z1 with a finite plate "
                 "distance. z1=%.6f, z2=%.6f.\n",
                 module_name, z1, z2);
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "FGM_DOUBLE_LAYER::Initial",
                                       error_string);
    }
    external_field_z = -(ep2 - ep1) / plate_distance;

    Nx = static_cast<int>(box_length.x);
    Ny = static_cast<int>(box_length.y);
    Nz = static_cast<int>(plate_distance);
    if (controller->Command_Exist(module_name, "Nx"))
    {
        controller->Check_Int(module_name, "Nx", "FGM_DOUBLE_LAYER::Initial");
        Nx = atoi(controller->Command(module_name, "Nx"));
    }
    if (controller->Command_Exist(module_name, "Ny"))
    {
        controller->Check_Int(module_name, "Ny", "FGM_DOUBLE_LAYER::Initial");
        Ny = atoi(controller->Command(module_name, "Ny"));
    }
    if (controller->Command_Exist(module_name, "Nz"))
    {
        controller->Check_Int(module_name, "Nz", "FGM_DOUBLE_LAYER::Initial");
        Nz = atoi(controller->Command(module_name, "Nz"));
    }
    if (Nx <= 0 || Ny <= 0 || Nz <= 0)
    {
        char error_string[CHAR_LENGTH_MAX];
        snprintf(error_string, sizeof(error_string),
                 "Reason:\n\t%s grid dimensions must be positive. "
                 "Nx=%d Ny=%d Nz=%d.\n",
                 module_name, Nx, Ny, Nz);
        controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                                       "FGM_DOUBLE_LAYER::Initial",
                                       error_string);
    }

    if (controller->Command_Exist(module_name, "FFT"))
    {
        use_fft = controller->Get_Bool(module_name, "FFT",
                                       "FGM_DOUBLE_LAYER::Initial");
    }
    if (controller->Command_Exist(module_name, "green_force_refresh_interval"))
    {
        controller->Check_Int(module_name, "green_force_refresh_interval",
                              "FGM_DOUBLE_LAYER::Initial");
        green_force_refresh_interval =
            atoi(controller->Command(module_name,
                                     "green_force_refresh_interval"));
    }
    if (controller->Command_Exist(module_name, "first_gamma"))
    {
        controller->Check_Float(module_name, "first_gamma",
                                "FGM_DOUBLE_LAYER::Initial");
        first_gamma = atof(controller->Command(module_name, "first_gamma"));
    }
    if (controller->Command_Exist(module_name, "second_gamma"))
    {
        controller->Check_Float(module_name, "second_gamma",
                                "FGM_DOUBLE_LAYER::Initial");
        second_gamma = atof(controller->Command(module_name, "second_gamma"));
    }
    if (controller->Command_Exist(module_name, "first_iteration_steps"))
    {
        controller->Check_Int(module_name, "first_iteration_steps",
                              "FGM_DOUBLE_LAYER::Initial");
        first_iteration_steps =
            atoi(controller->Command(module_name, "first_iteration_steps"));
    }
    if (controller->Command_Exist(module_name, "second_iteration_steps"))
    {
        controller->Check_Int(module_name, "second_iteration_steps",
                              "FGM_DOUBLE_LAYER::Initial");
        second_iteration_steps =
            atoi(controller->Command(module_name, "second_iteration_steps"));
    }
    if (controller->Command_Exist(module_name, "print_detail"))
    {
        print_detail = controller->Get_Bool(module_name, "print_detail",
                                            "FGM_DOUBLE_LAYER::Initial");
    }

    if (!controller->Command_Exist(module_name, "sphere_pos_file_name"))
    {
        char error_string[CHAR_LENGTH_MAX];
        snprintf(error_string, sizeof(error_string),
                 "Reason:\n\t%s requires sphere_pos_file_name.\n",
                 module_name);
        controller->Throw_SPONGE_Error(spongeErrorMissingCommand,
                                       "FGM_DOUBLE_LAYER::Initial",
                                       error_string);
    }
    strncpy(sphere_pos_file_name,
            controller->Command(module_name, "sphere_pos_file_name"),
            sizeof(sphere_pos_file_name) - 1);
    sphere_pos_file_name[sizeof(sphere_pos_file_name) - 1] = '\0';
    sphere_point_numbers =
        Count_Sphere_Sampling_Positions(controller, module_name,
                                        sphere_pos_file_name);

    image_layer_numbers = std::max(4, std::min(12, Nz / 8));

    Device_Malloc_Safely(reinterpret_cast<void**>(&d_atom_energy),
                         sizeof(float) * atom_numbers);
    Device_Malloc_Safely(reinterpret_cast<void**>(&d_total_energy),
                         sizeof(float));
    Reset_List(d_atom_energy, 0.0f, atom_numbers);

    controller->printf("    selected ions: [%d, %d)\n", ion_serial_start,
                       ion_serial_end);
    controller->printf("    z1=%.4f z2=%.4f ep1=%.4f ep2=%.4f\n", z1, z2, ep1,
                       ep2);
    controller->printf("    grid(Nx,Ny,Nz)=(%d,%d,%d), solver=%s\n", Nx, Ny,
                       Nz, use_fft ? "FFT" : "SOR");
    controller->printf("    sphere sampling file: %s (%d points)\n",
                       sphere_pos_file_name, sphere_point_numbers);
    controller->printf("    image layer numbers: %d\n", image_layer_numbers);

    controller->Step_Print_Initial(module_name, "%.2f");
    if (print_detail)
    {
        controller->Step_Print_Initial("FGM_green", "%.2f");
    }
    is_controller_printf_initialized = 1;
    is_initialized = 1;
}

void FGM_DOUBLE_LAYER::Clear()
{
    if (d_atom_energy != nullptr)
    {
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&d_atom_energy));
    }
    if (d_total_energy != nullptr)
    {
        Free_Single_Device_Pointer(reinterpret_cast<void**>(&d_total_energy));
    }
    controller = nullptr;
    is_initialized = 0;
}

void FGM_DOUBLE_LAYER::Update_Box(LTMatrix3 cell_, LTMatrix3 rcell_, LTMatrix3 g,
                                  float dt)
{
    if (!is_initialized)
    {
        return;
    }
    char error_string[CHAR_LENGTH_MAX];
    snprintf(error_string, sizeof(error_string),
             "Reason:\n\t%s does not support box updates in phase 1. "
             "dt=%.6f, g=(%.6f, %.6f, %.6f, %.6f, %.6f, %.6f).\n",
             module_name, dt, g.a11, g.a21, g.a22, g.a31, g.a32, g.a33);
    (void)cell_;
    (void)rcell_;
    if (controller != nullptr)
    {
        controller->Throw_SPONGE_Error(spongeErrorNotImplemented, module_name,
                                       error_string);
    }
    fprintf(stderr, "%s", error_string);
    std::exit(1);
}

void FGM_DOUBLE_LAYER::Compute_Green_Force_And_Energy(const VECTOR* crd,
                                                      const float* charge,
                                                      VECTOR* frc,
                                                      float* atom_energy)
{
    if (!is_initialized)
    {
        return;
    }
    Reset_List(d_atom_energy, 0.0f, atom_numbers);
    Launch_Device_Kernel(FGM_Reset_Selected_Atom_Energy,
                         (ion_numbers + CONTROLLER::device_max_thread - 1) /
                             CONTROLLER::device_max_thread,
                         CONTROLLER::device_max_thread, 0, nullptr, ion_numbers,
                         ion_serial_start, d_atom_energy);
    Launch_Device_Kernel(
        FGM_Image_Green_Force_And_Energy,
        (ion_numbers + CONTROLLER::device_max_thread - 1) /
            CONTROLLER::device_max_thread,
        CONTROLLER::device_max_thread, 0, nullptr, ion_numbers,
        ion_serial_start, ion_serial_end, crd, charge, box_length, z1,
        plate_distance, ep1, external_field_z, image_layer_numbers, frc,
        atom_energy, d_atom_energy);
}

void FGM_DOUBLE_LAYER::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized)
    {
        return;
    }
    Sum_Of_List(d_atom_energy, d_total_energy, atom_numbers);
    deviceMemcpy(&h_total_energy, d_total_energy, sizeof(float),
                 deviceMemcpyDeviceToHost);
    controller->Step_Print(module_name, h_total_energy);
    if (print_detail)
    {
        controller->Step_Print("FGM_green", h_total_energy);
    }
}
