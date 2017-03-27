#include "pso-mkl.h"
#include "pso.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "print_helpers.h"

constexpr bool minimum(double a1, double a2) {
  return a1 < a2;
}

double spherefunction(const pso::container_t<double>& x) {
  return (x * x).sum();
}

double ackleyfunction(const pso::container_t<double>& x) {
  const double n = double(x.size());
  const double first_sum =
      -20.0 * std::exp(-0.2 * std::sqrt((x * x).sum() / n));
  const double second_sum =
      -20.0 - M_E + std::exp(std::cos(2.0 * M_PI * x).sum() / n);
  return first_sum - second_sum;
}

double griewankfunction(const pso::container_t<double>& x) {
  const double sum = (x * x).sum() / 4000.0;
  double product = 1.0;
#pragma omp parallel for reduction(* : product)
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
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < x.size() - 1; ++i) {
    auto a = (x[i + 1] - x[i] * x[i]);
    auto b = (x[i] - 1);
    sum += 100 * a * a + b * b;
  }
  return sum;
}

void print_sstream_to_file_cout_clear(std::stringstream& ss,
                                      std::ofstream& file) {
  std::cout << ss.str();
  file << ss.str();
  ss.str("");
  ss.clear();
}

template <class PSOClass>
void unified_bounds_swarm_test(
    const std::string& test_name,
    const int& particle_size,
    const int& iterations_number,
    const int& tests_number,
    const int& dimensions,
    const double& bounds_low,
    const double& bounds_high,
    const pso::function_t<double>& function,
    const pso::predicate_t<double>& predicate = minimum,
    const std::string& name_prefix = "") {
  auto bounds = pso::unified_bounds(bounds_low, bounds_high, particle_size);
  std::stringstream output;
  std::ofstream test_file;
  test_file.open("tests\\" + test_name + ' ' + name_prefix + ".txt",
                 std::ofstream::out | std::ofstream::app);
  output << "~~~ Testing " << test_name << " ~~~\n Particles: " << particle_size
         << "\n Dimensions: " << dimensions
         << "\n Iterations: " << iterations_number << "\n\nRunning "
         << tests_number << " tests:\n";
  print_sstream_to_file_cout_clear(output, test_file);

  PSOClass particle_swarm(minimum, particle_size, bounds, function);
  pso::container_t<double> results(tests_number);
  auto pretty_int_width = 1;
  auto x = tests_number;
  while (x /= 10)
    pretty_int_width++;
  output << std::left << std::setprecision(5);

  // tests begin
  auto start_general = std::chrono::steady_clock::now();
  for (int i = 0; i < tests_number; ++i) {
    output << " Run # " << std::setw(pretty_int_width) << i + 1 << " = ";
    print_sstream_to_file_cout_clear(output, test_file);
    auto start = std::chrono::steady_clock::now();
    results[i] = particle_swarm(iterations_number).first;
    auto end = std::chrono::steady_clock::now();
    auto time_span = end - start;
    auto s =
        std::chrono::duration_cast<std::chrono::seconds>(time_span).count();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_span)
                  .count();
    output << std::setw(10) << results[i] << " Elapsed: " << s << "s "
           << ms - (s * 1000) << "ms\n";
    print_sstream_to_file_cout_clear(output, test_file);
  }
  auto end_general = std::chrono::steady_clock::now();
  // determine best
  double results_best = results.min();
  double results_worst = results.max();
  double results_average = results.sum() / tests_number;
  double std_err = sqrt(1 / double(tests_number - 1) *
                        std::pow((results - results_average), 2.0).sum()) /
                   sqrt(tests_number);
  // print results
  auto time_span_general = end_general - start_general;
  auto s_general =
      std::chrono::duration_cast<std::chrono::seconds>(time_span_general)
          .count();
  auto ms_general =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_span_general)
          .count();
  output << "\n\nResults:\n Elapsed: " << s_general << "s "
         << ms_general - (s_general * 1000) << "ms"
         << "\n Runs:    " << std::setw(10) << tests_number
         << "\n Best:    " << std::setw(10) << results_best
         << "\n Worst:   " << std::setw(10) << results_worst
         << "\n Average: " << std::setw(10) << results_average
         << "\n Std Err: " << std::setw(10) << std_err << '\n'
         << std::right << std::endl;
  print_sstream_to_file_cout_clear(output, test_file);
  test_file.close();
}

template <class PSOClass,
          int sphere_iterations,
          int ackley_iterations,
          int griewank_iterations,
          int rastrigin_iterations,
          int rosenbrock_iterations>
void whole_test(const std::string& test_class_name,
                int tests_number,
                const int particle_size) {
  std::cout << "** Testing class " << test_class_name << " ***\n";
  auto start = std::chrono::steady_clock::now();

  // sphere
  unified_bounds_swarm_test<PSOClass>(
      "sphere", particle_size, sphere_iterations, tests_number, 50, -100, 100,
      spherefunction, minimum, test_class_name);

  // ackley
  //unified_bounds_swarm_test<PSOClass>(
  //    "ackley", particle_size, ackley_iterations, tests_number, 20, -32.768,
  //    32.768, ackleyfunction, minimum, test_class_name);
  // griewank
  //unified_bounds_swarm_test<PSOClass>(
   //   "griewank", particle_size, griewank_iterations, tests_number, 50, -600,
  //    600, griewankfunction, minimum, test_class_name);
  // rastrigin
  //unified_bounds_swarm_test<PSOClass>(
  //    "rastrigin", particle_size, rastrigin_iterations, tests_number, 30, -5.12,
  //    5.12, rastriginfunction, minimum, test_class_name);
  // rosenbrock
  //unified_bounds_swarm_test<PSOClass>(
  //   "rosenbrock", particle_size, rosenbrock_iterations, tests_number, 30, -5,
  //    10, rosenbrockfunction, minimum, test_class_name);
  auto end = std::chrono::steady_clock::now();
  auto time_span = end - start;
  auto s = std::chrono::duration_cast<std::chrono::seconds>(time_span).count();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(time_span).count();
  std::cout << " Test complete in " << s << "s " << ms - (s * 1000) << "ms\n"
            << std::endl;
}

int _cdecl main() {
  std::cout << "Test begin.\n\n";
  // different functions have different tests
  auto tests_number = 10;
  auto particle_size = 30;
  whole_test<pso::ClassicGbestPSO<double>, 5000, 10000, 10000, 10000, 10000>(
      "with std random - gbest p" + std::to_string(particle_size), tests_number,
      particle_size);
  whole_test<pso::ClassicGbestMKLPSO, 5000, 10000, 10000, 10000, 10000>(
      "with MKL - gbest p" + std::to_string(particle_size), tests_number,
      particle_size);
  // whole_test<pso::ClassicLbestMKLPSO, 5000, 10000, 10000, 10000, 10000>(
  //    "with MKL - lbest p" + std::to_string(particle_size), tests_number,
  //    particle_size);
  std::cout << "Test end.\n";
  return 0;
}