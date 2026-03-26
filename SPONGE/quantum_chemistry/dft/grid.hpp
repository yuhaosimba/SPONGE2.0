#pragma once

#include "dft.hpp"
#include "lebedev.hpp"

struct QC_GRID_POINT
{
    float x;
    float y;
    float z;
    float w;
};

static inline float QC_Get_Covalent_Radius_Bohr(int Z)
{
    static const float rad_ang[19] = {1.00f,  // dummy
                                      0.31f, 0.28f, 1.28f, 0.96f, 0.84f, 0.76f,
                                      0.71f, 0.66f, 0.57f, 0.58f, 1.66f, 1.41f,
                                      1.21f, 1.11f, 1.07f, 1.05f, 1.02f, 1.06f};
    const float ang = (Z >= 1 && Z <= 18) ? rad_ang[Z] : 1.00f;
    return ang / 0.52917721092f;
}

static void QC_Gauss_Legendre_01(int n, std::vector<float>& nodes,
                                 std::vector<float>& weights)
{
    nodes.assign(n, 0.0f);
    weights.assign(n, 0.0f);
    const double eps = 1e-14;
    const int m = (n + 1) / 2;
    for (int i = 0; i < m; i++)
    {
        const double i1 = (double)i + 1.0;
        double z = cos(CONSTANT_Pi * (i1 - 0.25) / ((double)n + 0.5));
        double z1 = 0.0;
        double p1 = 0.0, p2 = 0.0, pp = 0.0;
        do
        {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 1; j <= n; j++)
            {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / (double)j;
            }
            pp = (double)n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (fabs(z - z1) > eps);
        const double x_low = -z;
        const double x_high = z;
        const double w = 2.0 / ((1.0 - z * z) * pp * pp);
        // Map [-1,1] -> [0,1]
        nodes[i] = (float)(0.5 * (x_low + 1.0));
        nodes[n - 1 - i] = (float)(0.5 * (x_high + 1.0));
        weights[i] = (float)(0.5 * w);
        weights[n - 1 - i] = (float)(0.5 * w);
    }
}

static void QC_Build_Fibonacci_Angular_Grid(int n_ang, std::vector<float>& dirs,
                                            std::vector<float>& w_ang)
{
    if (sponge_qc_lebedev::Load_Lebedev_Angular_Grid(n_ang, dirs, w_ang))
        return;
    dirs.assign((int)n_ang * 3, 0.0f);
    w_ang.assign(n_ang, 0.0f);
    const float golden = (sqrtf(5.0f) - 1.0f) * 0.5f;
    const float w = 4.0f * CONSTANT_Pi / (float)n_ang;
    for (int i = 0; i < n_ang; i++)
    {
        const float z = 1.0f - 2.0f * ((float)i + 0.5f) / (float)n_ang;
        const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi = 2.0f * CONSTANT_Pi *
                          ((float)i * golden - floorf((float)i * golden));
        dirs[(int)i * 3 + 0] = r * cosf(phi);
        dirs[(int)i * 3 + 1] = r * sinf(phi);
        dirs[(int)i * 3 + 2] = z;
        w_ang[i] = w;
    }
}

static inline __host__ __device__ float QC_Becke_Shape(float mu)
{
    // Becke smooth partition polynomial iteration.
    float x = mu;
    for (int it = 0; it < 3; it++)
    {
        x = 0.5f * x * (3.0f - x * x);
    }
    return 0.5f * (1.0f - x);
}

static inline void QC_Get_Atom_Coord_From_Env(const QC_MOLECULE& mol,
                                              const float* env, int atom_i,
                                              float& x, float& y, float& z)
{
    const int ptr = mol.h_atm[(int)atom_i * 6 + 1];
    x = env[ptr + 0];
    y = env[ptr + 1];
    z = env[ptr + 2];
}

static float QC_Atom_Partition_Weight(const QC_MOLECULE& mol, const float* env,
                                      int atom_i, float x, float y, float z)
{
    const int natm = mol.natm;
    std::vector<float> p((int)natm, 1.0f);
    for (int a = 0; a < natm; a++)
    {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        QC_Get_Atom_Coord_From_Env(mol, env, a, ax, ay, az);
        const float ra = sqrtf((x - ax) * (x - ax) + (y - ay) * (y - ay) +
                               (z - az) * (z - az));
        for (int b = 0; b < natm; b++)
        {
            if (a == b) continue;
            float bx = 0.0f, by = 0.0f, bz = 0.0f;
            QC_Get_Atom_Coord_From_Env(mol, env, b, bx, by, bz);
            const float rb = sqrtf((x - bx) * (x - bx) + (y - by) * (y - by) +
                                   (z - bz) * (z - bz));
            const float rab =
                sqrtf((ax - bx) * (ax - bx) + (ay - by) * (ay - by) +
                      (az - bz) * (az - bz));
            if (rab < 1e-10f) continue;
            float mu = (ra - rb) / rab;
            mu = fmaxf(-1.0f, fminf(1.0f, mu));
            p[a] *= QC_Becke_Shape(mu);
        }
    }
    float denom = 0.0f;
    for (int a = 0; a < natm; a++) denom += p[a];
    if (denom < 1e-20f) return 1.0f / (float)natm;
    return p[atom_i] / denom;
}

static std::vector<QC_GRID_POINT> QC_Build_Molecular_Grid(
    const QC_MOLECULE& mol, const float* env, const int* z_nuc, int z_stride,
    int n_radial, int n_angular)
{
    if (env == NULL || z_nuc == NULL || mol.natm <= 0) return {};

    std::vector<float> r_nodes, r_weights;
    QC_Gauss_Legendre_01(n_radial, r_nodes, r_weights);
    std::vector<float> dirs, w_ang;
    QC_Build_Fibonacci_Angular_Grid(n_angular, dirs, w_ang);

    std::vector<QC_GRID_POINT> grid;
    grid.reserve((int)mol.natm * n_radial * n_angular);
    for (int ia = 0; ia < mol.natm; ia++)
    {
        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
        QC_Get_Atom_Coord_From_Env(mol, env, ia, cx, cy, cz);
        const float scale = fmaxf(
            0.35f, QC_Get_Covalent_Radius_Bohr(z_nuc[(int)ia * z_stride]));
        for (int ir = 0; ir < n_radial; ir++)
        {
            const float x = r_nodes[ir];
            const float one_m_x = fmaxf(1e-8f, 1.0f - x);
            const float r = scale * x / one_m_x;
            const float drdx = scale / (one_m_x * one_m_x);
            const float wr = r_weights[ir] * r * r * drdx;
            for (int iang = 0; iang < n_angular; iang++)
            {
                const float gx = cx + r * dirs[(int)iang * 3 + 0];
                const float gy = cy + r * dirs[(int)iang * 3 + 1];
                const float gz = cz + r * dirs[(int)iang * 3 + 2];
                const float w_part =
                    QC_Atom_Partition_Weight(mol, env, ia, gx, gy, gz);
                const float w = wr * w_ang[iang] * w_part;
                if (fabsf(w) < 1e-18f) continue;
                grid.push_back({gx, gy, gz, w});
            }
        }
    }
    return grid;
}

void QUANTUM_CHEMISTRY::Update_DFT_Grid()
{
    std::vector<float> env_host(mol.h_env.size());
    if (mol.d_env != NULL)
        deviceMemcpy(env_host.data(), mol.d_env,
                     sizeof(float) * env_host.size(), deviceMemcpyDeviceToHost);
    else
        std::copy(mol.h_env.begin(), mol.h_env.end(), env_host.begin());

    std::vector<QC_GRID_POINT> grid =
        QC_Build_Molecular_Grid(mol, env_host.data(), mol.h_Z.data(), 1,
                                dft.dft_radial_points, dft.dft_angular_points);

    dft.max_grid_size = (int)grid.size();
    for (int i = 0; i < dft.max_grid_size; i++)
    {
        dft.h_grid_coords[(int)i * 3 + 0] = grid[i].x;
        dft.h_grid_coords[(int)i * 3 + 1] = grid[i].y;
        dft.h_grid_coords[(int)i * 3 + 2] = grid[i].z;
        dft.h_grid_weights[(int)i] = grid[i].w;
    }

    deviceMemcpy(dft.d_grid_coords, dft.h_grid_coords.data(),
                 sizeof(float) * dft.max_grid_size * 3,
                 deviceMemcpyHostToDevice);
    deviceMemcpy(dft.d_grid_weights, dft.h_grid_weights.data(),
                 sizeof(float) * dft.max_grid_size, deviceMemcpyHostToDevice);
}
