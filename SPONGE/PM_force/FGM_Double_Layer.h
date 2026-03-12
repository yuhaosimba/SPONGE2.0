#pragma once

#include "../common.h"
#include "../control.h"

struct FGM_DOUBLE_LAYER
{
    char module_name[CHAR_LENGTH_MAX];
    CONTROLLER* controller = nullptr;
    int is_initialized = 0;
    int is_controller_printf_initialized = 0;
    int last_modify_date = 20260312;

    bool print_detail = false;
    bool use_fft = true;

    int atom_numbers = 0;
    int ion_serial_start = 0;
    int ion_numbers = 0;
    int ion_serial_end = 0;

    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    int green_force_refresh_interval = 1;
    int image_layer_numbers = 8;

    float cutoff = 0.0f;
    float z1 = 0.0f;
    float z2 = 0.0f;
    float ep1 = 0.0f;
    float ep2 = 0.0f;
    float plate_distance = 0.0f;
    float external_field_z = 0.0f;
    float first_gamma = 1.9f;
    float second_gamma = 1.5f;
    int first_iteration_steps = 20;
    int second_iteration_steps = 80;

    int sphere_point_numbers = 0;
    char sphere_pos_file_name[CHAR_LENGTH_MAX];

    LTMatrix3 cell = {};
    LTMatrix3 rcell = {};
    VECTOR box_length = {};

    float* d_atom_energy = nullptr;
    float* d_total_energy = nullptr;
    float h_total_energy = 0.0f;

    static bool Is_Enabled_In_Controller(
        CONTROLLER* controller, const char* module_name = nullptr);

    void Initial(CONTROLLER* controller, int atom_numbers, LTMatrix3 cell,
                 LTMatrix3 rcell, VECTOR box_length, float cutoff,
                 const char* module_name = nullptr);
    void Clear();
    void Update_Box(LTMatrix3 cell, LTMatrix3 rcell, LTMatrix3 g, float dt);
    void Compute_Green_Force_And_Energy(const VECTOR* crd, const float* charge,
                                        VECTOR* frc, float* atom_energy);
    void Step_Print(CONTROLLER* controller);
};
