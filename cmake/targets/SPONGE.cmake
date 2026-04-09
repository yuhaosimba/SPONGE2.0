set(SPONGE_SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/main.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/common.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/control.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/xponge.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/MD_core/MD_core.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Domain_decomposition/Domain_decomposition.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/neighbor_list.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/full_neighbor_list.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/LJ_soft_core.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/solvent_LJ.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Middle_Langevin_MD.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Andersen_thermostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Berendsen_thermostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Bussi_thermostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Nose_Hoover_Chain.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/barostat/pressure_based_barostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/barostat/MC_barostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/PM_force/PM_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/angle/angle.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/angle/Urey_Bradley_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bond/bond.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/cmap/cmap.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/dihedral/dihedral.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/dihedral/improper_dihedral.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/nb14/nb14.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/restrain/restrain.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/constrain/constrain.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/constrain/settle.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/constrain/shake.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/virtual_atoms/virtual_atoms.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/collective_variable.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/RMSD.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/combine.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/simple_cv.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/tabulated.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/plugin/plugin.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/steer.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/restrain_cv.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/sinkmeta.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/SITS/SITS.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/custom_force/listed_forces.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/custom_force/pairwise_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/wall/soft_wall.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/wall/hard_wall.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/quantum_chemistry_init.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/quantum_chemistry_scf.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/quantum_chemistry_dft.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/quantum_chemistry_matrix.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/gpu_eri.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/sp/sp_kernels.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/md/md_kernels.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/integrals/eri/gpu/Rys/rys_kernels.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/sto-3g.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/3-21g.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/6-31g.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/6-31g_star.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/6-31g_starstar.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/6-311g.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/6-311g_star.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/pople/6-311g_starstar.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/def2/def2-svp.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/def2/def2-tzvp.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/def2/def2-tzvpp.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/def2/def2-qzvp.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/cc/cc-pvdz.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/basis/cc/cc-pvtz.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/NO_PBC/Coulomb_Force_No_PBC.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/NO_PBC/Lennard_Jones_force_No_PBC.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/NO_PBC/generalized_Born.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/sw.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/edip.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/eam.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/tersoff.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/eeq.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/bond_order.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/bond.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/vdw.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/over_under.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/valence_angle.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/torsion.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/hydrogen_bond.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_fixture.cpp)

set(SOURCES ${SPONGE_SOURCES})

find_package(tomlplusplus CONFIG REQUIRED)
add_library(
  sponge_toml STATIC
  ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml_decode.cpp)
set_source_files_properties(${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp
                            PROPERTIES LANGUAGE CXX)
set_source_files_properties(
  ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml_decode.cpp
  PROPERTIES LANGUAGE CXX)
target_include_directories(sponge_toml PUBLIC ${PROJECT_ROOT_DIR}/SPONGE)
target_link_libraries(sponge_toml PUBLIC tomlplusplus::tomlplusplus)

add_executable(${CURRENT_TARGET} ${SOURCES})
target_link_libraries(${CURRENT_TARGET} PRIVATE sponge_toml)
install(TARGETS ${CURRENT_TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(
  nbnxm_frozen_bench
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/frozen_port/nbnxm_frozen_bench.cu
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/frozen_port/ljewald_kernel_frozen_port.cu
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_exclusions.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_force_dump.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_fixture.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_live_builder_input.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_frozen_bench PRIVATE sponge_toml)
install(TARGETS nbnxm_frozen_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(
  nbnxm_stage_compare
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_stage_compare PRIVATE sponge_toml)
install(TARGETS nbnxm_stage_compare RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(
  nbnxm_builder_compare
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_exclusions.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_fixture.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_live_builder_input.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_builder_compare PRIVATE sponge_toml)
install(TARGETS nbnxm_builder_compare RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(
  nbnxm_stage_io_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_stage_io_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_stage_io_test PRIVATE sponge_toml)

add_executable(
  nbnxm_compare_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_compare_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_compare_test PRIVATE sponge_toml)

add_executable(
  nbnxm_builder_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_builder_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_builder_test PRIVATE sponge_toml)

add_executable(
  nbnxm_grid_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_grid_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_grid_test PRIVATE sponge_toml)

add_executable(
  nbnxm_pairlist_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_pairlist_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_pairlist_test PRIVATE sponge_toml)

add_executable(
  nbnxm_live_builder_input_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_live_builder_input_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_builder.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_exclusions.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_fixture.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_live_builder_input.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_compare.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_live_builder_input_test PRIVATE sponge_toml)

add_executable(
  nbnxm_force_dump_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_force_dump_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_force_dump.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_fixture.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp)
target_link_libraries(nbnxm_force_dump_test PRIVATE sponge_toml)

add_executable(
  nbnxm_layout_test
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/tests/nbnxm_layout_test.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_pairlist_types.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_fixture.cpp
  ${PROJECT_ROOT_DIR}/SPONGE/nbnxm/nbnxm_stage.cpp)
target_link_libraries(nbnxm_layout_test PRIVATE sponge_toml)
