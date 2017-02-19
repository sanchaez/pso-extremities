/* PSO interface */

#pragma once
#include <cmath>
#include <functional>
#include <random>
#include <valarray>

namespace pso {

using size_t = std::size_t;
template <typename value_t>
using container_t = std::valarray<value_t>;
template <typename value_t>
using dimension_limit_t = std::pair<value_t, value_t>;
template <typename value_t>
using comparing_predicate_t = std::function<bool(value_t, value_t)>;
template <typename value_t>
using function_t = std::function<value_t(container_t<value_t>)>;
template <typename value_t>
using result_t = std::pair<value_t, container_t<value_t>>;
template <typename value_t>
using dimention_container_t = container_t<dimension_limit_t<value_t>>;

template <typename value_t>
struct Particle {
  Particle(const container_t<value_t> &coordinates, const value_t &velocity)
      : x(coordinates), pbest(coordinates), v(velocity) {}
  value_t v;
  container_t<value_t> x;
  container_t<value_t> pbest;
};
template <typename value_t>
using particle_container_t = container_t<Particle<value_t>>;

template <typename value_t>
class PSOClassic {
 public:
  PSOClassic(const function_t& function,
             const dimention_container_t<value_t>& boudaries,
             const value_t& max_velocity)
      : vmax(max_velocity), bounds(boundaries), f(function) {}

  result_t<value_t> operator(const comparing_predicate_t& condition) {
    initialize_particles();

  }

 private:
  void initialize_particles(){};
  const value_t vmax;
  const dimention_container_t<value_t> bounds;
  const function_t f;
  container_t<value_t> gbest;
  particle_container_t<value_t> particles(50);
};
}