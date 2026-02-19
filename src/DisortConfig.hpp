#pragma once

#include "DisortTypes.hpp"
#include <optional>
#include <vector>

namespace disortpp {

/**
 * @brief Control flags for DISORT solver
 *
 */
struct DisortFlags {
  bool use_user_tau = false; ///< Return radiant quantities at user-specified optical depths
  bool use_user_mu = false; ///< Return radiant quantities at user-specified polar angles
  BoundaryConditionType ibcnd = BoundaryConditionType::General; ///< Boundary condition type
  bool use_lambertian_surface = true; ///< Isotropically reflecting bottom boundary
  bool use_thermal_emission = false; ///< Include thermal emission
  bool use_spherical_beam = false; ///< Pseudo-spherical geometry (otherwise plane-parallel)
  bool comp_only_fluxes = false; ///< Return only fluxes (no intensities)

  BrdfType brdf_type = BrdfType::None; ///< Type of BRDF
  bool intensity_corr_buras = false; ///< Apply intensity correction according to Buras & Emde 
  bool intensity_corr_nakajima = false; ///< Use original intensity correction routine from Nakajima & Tanaka (1988)
  bool use_delta_m_plus = false; ///< Use Delta-M+ scaling (Lin et al. 2018) instead of standard Delta-M
  bool output_fourier_expansion = false; ///< Return intensity_fourier_expansion as separate output
};

/**
 * @brief Boundary conditions for DISORT
 *
 */
struct BoundaryConditions {
  double direct_beam_flux = 0.0; ///< Intensity of incident parallel beam at top boundary
  double direct_beam_mu = 0.5; ///< Polar angle cosine of incident beam (positive)
  double direct_beam_phi = 0.0; ///< Azimuth angle of incident beam [0 to 360 deg]
  double isotropic_flux_top = 0.0; ///< Intensity of top-boundary isotropic illumination
  double isotropic_flux_bottom = 0.0; ///< Intensity of bottom-boundary isotropic illumination
  double temperature_top = 0.0; ///< Temperature [K] of top boundary
  double temperature_bottom = 0.0; ///< Temperature [K] of bottom boundary
  double emissivity_top = 1.0; ///< Emissivity of top boundary (needed if use_thermal_emission = TRUE)
  double surface_albedo = 0.0; ///< Albedo of bottom boundary (needed if use_lambertian_surface = TRUE)
};

/**
 * @brief RPV (Rahman-Pinty-Verstraete) BRDF specification
 *
 */
struct RpvBrdfSpec {
  double rho0 = 0.0;   ///< BRDF parameter rho0
  double k = 0.0;      ///< BRDF parameter k
  double theta = 0.0;  ///< BRDF parameter theta
  double sigma = 0.0;  ///< BRDF parameter sigma (for snow)
  double t1 = 0.0;     ///< BRDF parameter t1 (for snow)
  double t2 = 0.0;     ///< BRDF parameter t2 (for snow)
  double scale = 1.0;  ///< BRDF scaling factor
};

/**
 * @brief Ambrals BRDF specification
 *
 * Corresponds to ambrals_brdf_spec in original C code
 */
struct AmbralsBrdfSpec {
  double iso = 0.0;  ///< Isotropic component
  double vol = 0.0;  ///< Volumetric component
  double geo = 0.0;  ///< Geometric component
};

/**
 * @brief Cox & Munk BRDF specification
 *
 * Corresponds to cam_brdf_spec in original C code
 */
struct CoxMunkBrdfSpec {
  double u10 = 0.0;              ///< Wind speed at 10m [m/s]
  double pcl = 0.0;              ///< Pigment concentration (unused in native impl)
  double xsal = 0.0;             ///< Salinity (unused in native impl)
  double refractive_index = 1.34; ///< Refractive index of water (default seawater)
  bool do_shadow = true;          ///< Enable Tsang shadowing effect
};

/**
 * @brief Hapke BRDF specification
 *
 * Parameters for the Hapke opposition-effect surface reflectance model.
 * Reference: Hapke (1993), "Theory of Reflectance and Emittance Spectroscopy", Eq. 8.89
 */
struct HapkeBrdfSpec {
  double b0 = 1.0;   ///< Opposition effect amplitude
  double hh = 0.06;  ///< Opposition effect angular width
  double w  = 0.6;   ///< Single scattering albedo of surface particles
};

/**
 * @brief BRDF specification container
 *
 */
struct BrdfSpecification {
  std::optional<RpvBrdfSpec>     rpv;      ///< RPV BRDF (if used)
  std::optional<AmbralsBrdfSpec> ambrals;  ///< Ambrals/Ross-Li BRDF (if used)
  std::optional<CoxMunkBrdfSpec> cox_munk; ///< Cox & Munk BRDF (if used)
  std::optional<HapkeBrdfSpec>   hapke;    ///< Hapke BRDF (if used; defaults apply if unset)

  void setRpv(const RpvBrdfSpec& spec)          { rpv      = spec; }
  void setAmbrals(const AmbralsBrdfSpec& spec)   { ambrals  = spec; }
  void setCoxMunk(const CoxMunkBrdfSpec& spec)   { cox_munk = spec; }
  void setHapke(const HapkeBrdfSpec& spec)       { hapke    = spec; }
};

/**
 * @brief Main DISORT configuration/input state
 *
 * This class holds all input parameters for the DISORT solver.
 */
class DisortConfig {
public:
  // Flags and boundary conditions
  DisortFlags flags;
  BoundaryConditions bc;
  BrdfSpecification brdf;

  // Dimensions
  int num_layers = 0;  ///< Number of computational layers
  int num_phase_func_moments = 0;  ///< Number of phase function moments (not including zeroth)
  int num_streams = 0;  ///< Number of streams (computational polar angles, even and >= 4)
  int num_user_tau = 0;  ///< Number of computational optical depths
  int num_user_mu = 0;  ///< Number of user polar angles
  int num_phi = 0;  ///< Number of azimuthal angles for intensity output
  int num_phase_func_angles = 0;  ///< Number of phase function angle grid points

  /// Derived: max(num_phase_func_moments, num_streams) — used for phaseFunctionMoments array sizing
  int nmomNstr() const { return std::max(num_phase_func_moments, num_streams); }

  // Physical parameters
  double wavenumber_low = 0.0;  ///< Wavenumber lower bound [cm^-1] for Planck function
  double wavenumber_high = 0.0;  ///< Wavenumber upper bound [cm^-1] for Planck function
  double accuracy_fourier_series = 0.0;   ///< Convergence criterion for azimuthal series
  double bottom_radius = 0.0;  ///< Planet bottom_radius (same units as level_altitudes, used if use_spherical_beam=true)

  // Layer properties (0-based indexing)
  std::vector<double> delta_tau;    ///< Optical depth of each layer [num_layers]
  std::vector<double> single_scat_albedo;    ///< Single-scatter surface_albedo of each layer [num_layers]
  std::vector<std::vector<double>> phase_function_moments;  ///< Phase function moments [num_layers][nmomNstr()+1]
  
  //Level properties
  std::vector<double> temperature;     ///< Temperature at levels [num_layers+1] (if use_thermal_emission=true)
  std::vector<double> level_altitudes; ///< Altitude at levels [num_layers+1], Spherical geometry (if use_spherical_beam=true)

  // User output specification
  std::vector<double> tau_user;     ///< User optical depths [num_user_tau]
  std::vector<double> mu_user;      ///< User polar angle cosines [num_user_mu]
  std::vector<double> phi_user;      ///< User azimuthal angles [deg] [num_phi]

  // Phase function specification (if using new intensity correction)
  std::vector<double> mu_phase_function; ///< Scattering angle values [num_phase_func_angles]
  std::vector<std::vector<double>> phase_function;    ///< Phase function values [num_layers][num_phase_func_angles]

  /**
   * @brief Default constructor
   */
  DisortConfig() = default;

  /**
   * @brief Construct with basic dimensions
   * @param nlyr_ Number of layers
   * @param nstr_ Number of streams
   * @param nmom_ Number of phase function moments
   */
  DisortConfig(int nlyr_, int nstr_, int nmom_ = 0);

  /**
   * @brief Allocate arrays based on current dimensions
   *
   * Allocates storage for all dynamic arrays based on num_layers, num_streams, etc.
   */
  void allocate();

  /**
   * @brief Validate configuration
   * @throws std::invalid_argument if configuration is invalid
   *
   */
  void validate() const;

  /**
   * @brief Access phase_function_moments as 2D array
   * @param k Moment index (0-based: 0..nmomNstr())
   * @param lc Layer index (0-based: 0..num_layers-1)
   * @return Reference to phase_function_moments element
   */
  double& phaseFunctionMoments(int k, int lc) {
    return phase_function_moments[lc][k];
  }

  const double& phaseFunctionMoments(int k, int lc) const {
    return phase_function_moments[lc][k];
  }

  /**
   * @brief Fill phase_function_moments with Henyey-Greenstein moments: phaseFunctionMoments(k) = g^k
   * @param g Asymmetry parameter in (-1, 1)
   * @param lc Layer index (0-based), or -1 to fill all layers
   */
  void setHenyeyGreenstein(double g, int lc = -1);

  /**
   * @brief Fill phase_function_moments with isotropic phase function: phaseFunctionMoments(0)=1, rest=0
   * @param lc Layer index (0-based), or -1 to fill all layers
   */
  void setIsotropic(int lc = -1);

  /**
   * @brief Fill phase_function_moments with Rayleigh phase function: phaseFunctionMoments(0)=1, phaseFunctionMoments(2)=0.1, rest=0
   * @param lc Layer index (0-based), or -1 to fill all layers
   */
  void setRayleigh(int lc = -1);

  /**
   * @brief Fill phase_function_moments with Haze-L phase function (Garcia & Siewert 1985, Table 12-16)
   * @param lc Layer index (0-based), or -1 to fill all layers
   */
  void setHazeGarciaSiewert(int lc = -1);

  /**
   * @brief Fill phase_function_moments with Cloud C.1 phase function (Garcia & Siewert 1985, Table 19-20)
   * @param lc Layer index (0-based), or -1 to fill all layers
   */
  void setCloudGarciaSiewert(int lc = -1);

  /**
   * @brief Fill phase_function_moments using a named phase function type (equivalent to c_getmom())
   * @param type Phase function type
   * @param g Asymmetry parameter (only used for HenyeyGreenstein)
   * @param lc Layer index (0-based), or -1 to fill all layers
   */
  void setPhaseFunction(PhaseFunction type, double g = 0.0, int lc = -1);
};

} // namespace disortpp
