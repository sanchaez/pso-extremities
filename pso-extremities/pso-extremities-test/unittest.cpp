#include "pso.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include "print_helpers.h"

constexpr bool minimum(double a1, double a2) {
  return a1 < a2;
}

double spherefunction(const pso::container_t<double>& x) {
  return (x * x).sum();
}

double ackleyfunction(const pso::container_t<double>& x) {
  const double d_inv = 1.0/ double(x.size());
  const double a = 20.0;
  return (-a * std::exp(-0.2 * std::sqrt(d_inv * (x * x).sum())) -
          std::exp(d_inv * std::cos(M_PI * 2.0 * x).sum()) + a + M_E);
}

double griewankfunction(const pso::container_t<double>& x) {
  const double sum = (x*x).sum() / 4000.0;
  double product = 1.0;
#pragma omp parallel for reduction(*:product)
  for (int i = 0; i < x.size(); ++i) {
    product *= std::cos(x[i] / double(i + 1));
  }
  return sum - product + 1;
}

double rastriginfunction(const pso::container_t<double>& x) {
  return 10.0 * x.size() + (x * x - 10.0 * std::cos(2 * M_PI * x)).sum();
}

double rosenbrockfunction(const pso::container_t<double> x) {
  double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < x.size() - 1; ++i) {
    auto c = (x[i + 1] - x[i] * x[i]);
    sum += 100 * c * c + (x[i] - 1) * (x[i] - 1);
  }
  return sum;
}

void unified_bounds_swarm_test(
    const std::string& test_name,
    const int& particle_size,
    const int& iterations_number,
    const int& tests_number,
    const int& dimensions,
    const double& bounds_low,
    const double& bounds_high,
    const pso::function_t<double>& function,
    const pso::predicate_t<double>& predicate = minimum) {
  auto bounds = pso::unified_bounds(bounds_low, bounds_high, particle_size);
  std::cout << "~~~ Testing " << test_name
            << " ~~~\n Particles: " << particle_size
            << "\n Dimensions: " << dimensions
            << "\n Iterations: " << iterations_number << "\n\nRunning tests:\n";

  pso::PSOClassic<double> particle_swarm(minimum, particle_size, bounds,
                                         function);

  pso::container_t<double> results(tests_number);
  for (int i = 0; i < tests_number; ++i) {
    std::cout << " Run # " << i + 1 << " = " ;
    results[i] = particle_swarm(iterations_number).first;
    std::cout << results[i] << '\n';
  }
  // determine best
  double results_best = results.min();
  double results_worst = results.max();
  double results_average = results.sum() / tests_number;
  double std_err = sqrt(1 / double(tests_number - 1) *
                        std::pow((results - results_average), 2.0).sum()) /
                   sqrt(tests_number);
  std::cout << "\n\nResults:"
            << "\n Runs: " << tests_number << "\n Best: " << results_best
            << "\n Worst: " << results_worst
            << "\n Average: " << results_average << "\n Std Err: " << std_err
            << '\n'
            << std::endl;
}

int main() {
  std::cout << "Test begin.\nNote: some functions can be very slow!\n\n";
  // different functions have different tests
  auto tests_number = 30;
  auto iterations_number = 10000;
  auto particle_size = 50;
  // sphere
  unified_bounds_swarm_test("sphere", particle_size, iterations_number,
                            tests_number, 50, -100, 100, spherefunction);
  // ackley
  unified_bounds_swarm_test("ackley", particle_size, iterations_number,
                            tests_number, 20, -32.768, 32.768, ackleyfunction);
  // griewank
  unified_bounds_swarm_test("griewank", particle_size, iterations_number,
                            tests_number, 50, -600, 600, griewankfunction);
  // rastrigin
  unified_bounds_swarm_test("rastrigin", particle_size, iterations_number,
                            tests_number, 30, -5.12, 5.12, rastriginfunction);
  // rosenbrock
  unified_bounds_swarm_test("rosenbrock", particle_size, iterations_number,
                            tests_number, 30, -5, 10, rosenbrockfunction);
  std::cout << "Test end.\n";
  std::cin.get();
}