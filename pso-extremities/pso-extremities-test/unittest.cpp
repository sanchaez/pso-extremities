#include "pso.h"
#define _USE_MATH_DEFINES
#include <math.h>

constexpr bool minimum(double a1, double a2) {
  return a1 < a2;
}

double spherefunction(pso::container_t<double> x) {
  return (x * x).sum();
}

double ackleyfunction(pso::container_t<double> x) {
  const double d = double(x.size());
  const double a = 20.0;
  const double b = 0.2;
  const double c = M_PI * 2.0;
  const double cubic_sum = std::pow(x, 2.0).sum();
  const double first_arg = a * std::exp(-b * std::sqrt((1.0 / d) * cubic_sum));
  const double cos_sum = std::cos(c * x).sum();
  const double second_arg = std::exp(1.0 / d * cos_sum);
  return (-first_arg - second_arg + a + M_E);
}



bool is_true(const std::valarray<bool>& bool_result) {
  for (auto item : bool_result) {
    if (!item) {
      return false;
    }
  }
  return true;
}

in
  auto test_particle_size = 50;
  auto test_dimensions = 20;
  auto test_bounds = pso::unified_bounds<double>(-32, 32, test_dimensions);

  pso::PSOClassic<double> test_particle_swarm(minimum, test_particle_size,
                                              test_bounds, ackleyfunction);
  SECTION("Creating PSOClassic object") {
    REQUIRE(test_particle_swarm.particles_number() == test_particle_size);
    REQUIRE(test_particle_swarm.dimensions_number() == test_dimensions);
    REQUIRE(test_particle_swarm.predicate()(0, 1) == minimum(0, 1));
    REQUIRE(is_true(test_particle_swarm.bounds() == test_bounds));
  }
  SECTION("Testing operator") {
    auto iterations_number = 30000;
    auto x = test_particle_swarm(iterations_number);
    INFO("Value :");
    CAPTURE(x.first);
    INFO("Coordinates :");
    for (const auto & x : x.second) {
      CAPTURE(x);
    }
    auto result_test = ackleyfunction(pso::container_t<double>(0.0, test_dimensions));
    CAPTURE(result_test);
    REQUIRE(test_particle_swarm.function()(pso::container_t<double>(10, 10)) ==
            spherefunction(pso::container_t<double>(10, 10)));
    REQUIRE(abs(x.first - result_test) < 1e-15 );

  }
  
}