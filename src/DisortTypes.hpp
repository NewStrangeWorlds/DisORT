#pragma once

namespace disortpp {

// Phase function types
enum class PhaseFunction {
  Isotropic = 1,
  Rayleigh = 2,
  HenyeyGreenstein = 3,
  HazeGarciaSiewert = 4,
  CloudGarciaSiewert = 5
};

// Boundary condition types
enum class BoundaryConditionType {
  General = 0,  // GENERAL_BC
  Special = 1   // SPECIAL_BC (surface_albedo/transmissivity only)
};

// Illumination types
enum class IlluminationType {
  Top = 1,  // TOP_ILLUM
  Bottom = 2  // BOT_ILLUM
};

// Message types
enum class MessageType {
  Warning = 0,  // DS_WARNING
  Error = 1     // DS_ERROR
};

// Verbosity levels
enum class VerbosityLevel {
  Verbose = 0,  // VERBOSE
  Quiet = 1     // QUIET
};

// BRDF types
enum class BrdfType {
  None = 0,   // BRDF_NONE
  RPV = 1,    // BRDF_RPV (Rahman-Pinty-Verstraete)
  CoxMunk = 2,  // BRDF_CAM (Cox & Munk)
  Ambrals = 3,  // BRDF_AMB
  Hapke = 4     // BRDF_HAPKE
};

// Mathematical constants (use C++ standard library)
constexpr double DEG = 3.14159265358979323846 / 180.0;  // M_PI / 180

// NMUG: Number of angle cosine quadrature points for integrating BRDF
constexpr int NMUG = 200;

} // namespace disortpp
