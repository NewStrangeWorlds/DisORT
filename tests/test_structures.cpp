#include "testing.hpp"
#include "DisortConfig.hpp"
#include "DisortResult.hpp"

using namespace disortpp;

// Test DisortConfig
TEST(DisortConfigTest, Construction) {
  DisortConfig config(5, 16, 32);

  EXPECT_EQ(config.num_layers, 5);
  EXPECT_EQ(config.num_streams, 16);
  EXPECT_EQ(config.num_phase_func_moments, 32);
  EXPECT_EQ(config.nmomNstr(), 32);  // max(16, 32)
}

TEST(DisortConfigTest, InvalidDimensions) {
  // num_layers must be positive
  EXPECT_THROW(DisortConfig(-1, 16, 0), std::invalid_argument);

  // num_streams must be even
  EXPECT_THROW(DisortConfig(5, 15, 0), std::invalid_argument);

  // num_streams must be >= 4
  EXPECT_THROW(DisortConfig(5, 0, 0), std::invalid_argument);
  EXPECT_THROW(DisortConfig(5, 2, 0), std::invalid_argument);
}

TEST(DisortConfigTest, Allocation) {
  DisortConfig config(5, 16, 32);
  config.allocate();

  // Check that arrays are allocated
  EXPECT_EQ(config.delta_tau.size(), 5);
  EXPECT_EQ(config.single_scat_albedo.size(), 5);
  EXPECT_EQ(config.phase_function_moments.size(), 5);       // num_layers layers
  EXPECT_EQ(config.phase_function_moments[0].size(), 33);  // nmomNstr()+1 moments per layer
}

TEST(DisortConfigTest, AllocationWithPlanck) {
  DisortConfig config(5, 16, 32);
  config.flags.use_thermal_emission = true;
  config.allocate();

  // Temperature array should be allocated (num_layers+1 elements)
  EXPECT_EQ(config.temperature.size(), 6);
}

TEST(DisortConfigTest, PMOMAccess) {
  DisortConfig config(3, 8, 10);
  config.allocate();

  // Test phaseFunctionMoments(k, lc) accessor (0-based layer indexing)
  config.phaseFunctionMoments(1, 0) = 1.5;
  config.phaseFunctionMoments(5, 1) = 2.5;

  EXPECT_DOUBLE_EQ(config.phaseFunctionMoments(1, 0), 1.5);
  EXPECT_DOUBLE_EQ(config.phaseFunctionMoments(5, 1), 2.5);
}

TEST(DisortConfigTest, Validation) {
  DisortConfig config(5, 16, 32);
  config.allocate();

  config.flags.comp_only_fluxes = true;

  // Set valid values
  for (int lc = 0; lc < config.num_layers; ++lc) {
    config.delta_tau[lc] = 0.1 * lc;
    config.single_scat_albedo[lc] = 0.9;
  }
  config.accuracy_fourier_series = 1e-4;

  // Should not throw
  EXPECT_NO_THROW(config.validate());
}

TEST(DisortConfigTest, ValidationFailsOnNegativeDtauc) {
  DisortConfig config(5, 16, 32);
  config.allocate();

  config.delta_tau[0] = -0.1;  // Invalid
  config.single_scat_albedo[0] = 0.9;
  config.accuracy_fourier_series = 1e-4;

  EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(DisortConfigTest, ValidationFailsOnInvalidSsalb) {
  DisortConfig config(5, 16, 32);
  config.allocate();

  config.delta_tau[0] = 0.1;
  config.single_scat_albedo[0] = 1.5;  // Invalid (> 1.0)
  config.accuracy_fourier_series = 1e-4;

  EXPECT_THROW(config.validate(), std::invalid_argument);
}

// Test DisortResult
TEST(DisortResultTest, Allocation) {
  DisortResult result;

  int num_user_tau = 10;
  int num_user_mu = 5;
  int num_phi = 3;
  int num_streams = 16;

  result.allocate(num_user_tau, num_user_mu, num_phi, num_streams, false, true, false);

  // Check sizes
  EXPECT_EQ(result.flux_direct_beam.size(), num_user_tau);
  EXPECT_EQ(result.intensity.size(), num_user_tau);
  EXPECT_EQ(result.intensity[0].size(), num_user_mu);
  EXPECT_EQ(result.intensity[0][0].size(), num_phi);
  EXPECT_EQ(result.intensity_azimuth_avg.size(), num_user_tau);
  EXPECT_EQ(result.intensity_azimuth_avg[0].size(), num_user_mu);
}

TEST(DisortResultTest, AllocationSpecialBC) {
  DisortResult result;

  int num_user_tau = 10;
  int num_user_mu = 5;
  int num_phi = 3;
  int num_streams = 16;

  result.allocate(num_user_tau, num_user_mu, num_phi, num_streams, true, false, false);

  // For SPECIAL_BC mode
  EXPECT_EQ(result.albedo_medium.size(), num_user_mu);
  EXPECT_EQ(result.transmissivity_medium.size(), num_user_mu);
}

TEST(DisortResultTest, AllocationWithUUM) {
  DisortResult result;

  int num_user_tau = 10;
  int num_user_mu = 5;
  int num_phi = 3;
  int num_streams = 16;

  result.allocate(num_user_tau, num_user_mu, num_phi, num_streams, false, true, true);

  // intensity_fourier_expansion array should be allocated
  EXPECT_EQ(result.intensity_fourier_expansion.size(), num_user_tau);
  EXPECT_EQ(result.intensity_fourier_expansion[0].size(), num_user_mu);
  EXPECT_EQ(result.intensity_fourier_expansion[0][0].size(), num_streams + 1);
}

TEST(DisortResultTest, RadiantQuantitiesInitialization) {
  DisortResult result;

  int num_user_tau = 5;
  result.allocate(num_user_tau, 1, 1, 16, false, false, false);

  // All radiant quantities should be initialized to zero
  for (int lu = 0; lu < num_user_tau; ++lu) {
    EXPECT_DOUBLE_EQ(result.flux_direct_beam[lu], 0.0);
    EXPECT_DOUBLE_EQ(result.flux_down[lu], 0.0);
    EXPECT_DOUBLE_EQ(result.flux_up[lu], 0.0);
    EXPECT_DOUBLE_EQ(result.flux_tau_divergence[lu], 0.0);
  }
}

// Test structures (Flags, BoundaryConditions, BRDF)
TEST(StructuresTest, DisortFlagsDefaults) {
  DisortFlags flags;

  EXPECT_FALSE(flags.use_user_tau);
  EXPECT_FALSE(flags.use_user_mu);
  EXPECT_TRUE(flags.use_lambertian_surface);
  EXPECT_FALSE(flags.use_thermal_emission);
  EXPECT_FALSE(flags.use_spherical_beam);
  EXPECT_FALSE(flags.comp_only_fluxes);
}

TEST(StructuresTest, BoundaryConditionsDefaults) {
  BoundaryConditions bc;

  EXPECT_DOUBLE_EQ(bc.direct_beam_flux, 0.0);
  EXPECT_DOUBLE_EQ(bc.direct_beam_mu, 0.5);
  EXPECT_DOUBLE_EQ(bc.direct_beam_phi, 0.0);
  EXPECT_DOUBLE_EQ(bc.isotropic_flux_top, 0.0);
  EXPECT_DOUBLE_EQ(bc.isotropic_flux_bottom, 0.0);
}

TEST(StructuresTest, BrdfSpecification) {
  BrdfSpecification brdf;

  // Initially empty
  EXPECT_FALSE(brdf.rpv.has_value());
  EXPECT_FALSE(brdf.ambrals.has_value());
  EXPECT_FALSE(brdf.cox_munk.has_value());

  // Set RPV
  RpvBrdfSpec rpv_spec;
  rpv_spec.rho0 = 0.5;
  rpv_spec.k = 0.8;

  brdf.setRpv(rpv_spec);
  ASSERT_TRUE(brdf.rpv.has_value());
  EXPECT_DOUBLE_EQ(brdf.rpv->rho0, 0.5);
  EXPECT_DOUBLE_EQ(brdf.rpv->k, 0.8);
}

