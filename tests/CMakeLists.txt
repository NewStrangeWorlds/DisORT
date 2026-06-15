# Test executable (no external test framework required)
add_executable(disortpp_tests
    main.cpp
    test_basic.cpp
    test_structures.cpp
    test_utils.cpp
    test_linear_algebra.cpp
    test_solver.cpp
    test_flux_solver.cpp
    test_jacobian.cpp
)

target_include_directories(disortpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(disortpp_tests
    disortpp
    Eigen3::Eigen
)

# Register a single CTest entry that runs the whole suite
add_test(NAME disortpp_tests COMMAND disortpp_tests)
set_tests_properties(disortpp_tests PROPERTIES PASS_REGULAR_EXPRESSION "tests passed")
