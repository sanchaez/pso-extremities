#include "external/catch.hpp"
#include "pso.h"
constexpr bool minimum(double a1, double a2) {
  return a1 < a2;
}

double spherefunction(pso::container_t<double> x) {
  return (x * x).sum();
}

bool is_true(const std::valarray<bool>& bool_result) {
  for (auto item : bool_result) {
    if (!item) {
      return false;
    }
  }
  return true;
}

TEST_CASE("Dummy", "[dummy]") {
  auto test_particle_size = 50;
  auto test_dimensions = 50;
  auto test_bounds = pso::unified_bounds<double>(-100, 100, test_dimensions);

  pso::PSOClassic<double> test_particle_swarm(minimum, test_particle_size,
                                              test_bounds);
  SECTION("Creating PSOClassic object") {
    REQUIRE(test_particle_swarm.particles_number() == test_particle_size);
    REQUIRE(test_particle_swarm.dimensions_number() == test_dimensions);
    REQUIRE(test_particle_swarm.predicate()(0, 1) == minimum(0, 1));
    REQUIRE(is_true(test_particle_swarm.bounds() == test_bounds));
  }
  SECTION("Testing operator") {
    auto iterations_number = test_dimensions * 10000;
    auto x = test_particle_swarm(iterations_number, spherefunction);
    INFO("Value :");
    CAPTURE(x.first);
    INFO("Coordinates :");
    for (const auto & x : x.second) {
      CAPTURE(x);
    }
    auto result_test = spherefunction(pso::container_t<double>(0.0, test_dimensions));
    CAPTURE(result_test);
    REQUIRE(test_particle_swarm.function()(pso::container_t<double>(10, 10)) ==
            spherefunction(pso::container_t<double>(10, 10)));
    REQUIRE(abs(x.first - result_test) < 1e-1 );

  }
  
}