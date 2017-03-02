/* PSO interface */

#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <valarray>
#include "print_helpers.h"

namespace pso {
using generator_t = std::mt19937;
template <typename value_t>
using container_t = std::valarray<value_t>;
template <typename value_t>
using dimension_limit_t = std::pair<value_t, value_t>;
template <typename value_t>
using dimention_container_t = container_t<dimension_limit_t<value_t>>;
template <typename value_t>
using predicate_t = std::function<bool(value_t, value_t)>;
template <typename value_t>
using function_t = std::function<value_t(container_t<value_t>)>;
template <typename value_t>
using value_coordinates_t = std::pair<value_t, container_t<value_t>>;
template <typename value_t>
using uniform_distribution_t = std::uniform_real_distribution<value_t>;

template <typename value_t>
dimention_container_t<value_t> unified_bounds(
    const dimension_limit_t<value_t>& limit,
    const int size) {
  return dimention_container_t<value_t>(limit, size);
}
template <typename value_t>
dimention_container_t<value_t> unified_bounds(const value_t left_bound,
                                              const value_t right_bound,
                                              const int size) {
  return dimention_container_t<value_t>(std::make_pair(left_bound, right_bound),
                                        size);
}

template <typename value_t>
struct Particle {
  Particle() : v(0), x(0), best(0) {}
  Particle(const container_t<value_t>& coordinates, const value_t& velocity)
      : x(coordinates), best(coordinates), v(velocity) {}
  container_t<value_t> v;
  container_t<value_t> x;
  container_t<value_t> best;
};

template <typename value_t>
using particle_container_t = container_t<Particle<value_t>>;
/// Abstract class for all particle swarms
template <typename value_t>
class AbstractPSO {
 public:
  AbstractPSO(const predicate_t<value_t>& predicate,
              const int particles_number,
              const dimention_container_t<value_t>& boundaries,
              const function_t<value_t>& function)
      : m_bounds(boundaries),
        m_compare(predicate),
        m_dimensions_number(boundaries.size()),
        m_particles_number(particles_number),
        m_particles(particles_number),
        m_function(function) {}

  const int particles_number() { return m_particles_number; }
  const int dimensions_number() { return m_dimensions_number; }
  const function_t<value_t> function() { return m_function; }
  const predicate_t<value_t> predicate() { return m_compare; }
  const dimention_container_t<value_t> bounds() { return m_bounds; }

  void set_predicate(const predicate_t<value_t>& predicate) {
    m_compare = predicate;
  }
  void set_function(const function_t<value_t>& function) {
    m_function = function;
  }
  void set_bounds(const dimention_container_t<value_t>& bounds) {
    m_bounds = bounds;
    m_dimensions_number = bounds.size();
  }
  void set_particles_number(const int& particles_number) {
    m_particles_number = particles_number;
  }

  void initialize_particles() {
    m_particles.resize(m_particles_number);
// init every dimension separately
#pragma omp parallel for
    for (int particle = 0; particle < m_particles_number; ++particle) {
      m_particles[particle].x.resize(m_dimensions_number);
      m_particles[particle].v.resize(m_dimensions_number);
      m_particles[particle].best.resize(m_dimensions_number);
#pragma omp parallel for
      for (int dimension = 0; dimension < m_dimensions_number; ++dimension) {
        uniform_distribution_t<value_t> dimension_distribution(
            m_bounds[dimension].first, m_bounds[dimension].second);
        uniform_distribution_t<value_t> velocity_distribution(
            0, m_bounds[dimension].second);
        m_particles[particle].x[dimension] =
            dimension_distribution(m_generator);
        m_particles[particle].v[dimension] = velocity_distribution(m_generator);
        m_particles[particle].best[dimension] =
            m_particles[particle].x[dimension];
      }
    }
  }

  virtual value_coordinates_t<value_t> operator()(const int iterations_max) = 0;

 protected:
  container_t<value_t> update_function(const value_t& eps1,
                                       const value_t& eps2) {}
  int m_particles_number;
  int m_dimensions_number;
  function_t<value_t> m_function;
  predicate_t<value_t> m_compare;
  generator_t m_generator;
  container_t<value_t> m_gbest;

  dimention_container_t<value_t> m_bounds;
  particle_container_t<value_t> m_particles;
};

template <typename value_t>
class PSOClassic : public AbstractPSO<value_t> {
 public:
  using AbstractPSO::AbstractPSO;
  value_coordinates_t<value_t> operator()(const int iterations_max) {
    initialize_particles();
    for (int i = 0; i < iterations_max; ++i) {
#pragma omp parallel for
      for (int i = 0; i < m_particles_number; ++i) {
        m_particles[i] = update_particle(m_particles[i]);
      }
      update_gbest();
      if (!(i % 1000)) {
        std::cout << "On iteration " << i
                  << " value is: " << m_function(m_gbest) << std::endl;
      }
    }
    return std::make_pair(m_function(m_gbest), m_gbest);
  }

  void initialize_particles() {
    AbstractPSO<value_t>::initialize_particles();
    // init gbest
    m_gbest = m_particles[0].x;
    update_gbest();
  }

 private:
  Particle<value_t> update_particle(const Particle<value_t>& p) {
    Particle<value_t> new_particle = p;
    container_t<value_t> eps1(m_dimensions_number);
    container_t<value_t> eps2(m_dimensions_number);

    uniform_distribution_t<value_t> eps_distribution(0, 1);
    eps_distribution.reset();
#pragma omp parallel for
    for (int i = 0; i < m_dimensions_number; ++i) {
      eps1[i] = eps_distribution(m_generator);
    }
    eps_distribution.reset();
#pragma omp parallel for
    for (int i = 0; i < m_dimensions_number; ++i) {
      eps2[i] = eps_distribution(m_generator);
    }

    container_t<value_t> new_velocity =
        value_t(0.72984) *
        (p.v + (p.best - p.x) * eps1 * 2.05 + (m_gbest - p.x) * eps2 * 2.05);
// floor velocity to lowest value

#pragma omp parallel for
    for (int dimension = 0; dimension < m_dimensions_number; ++dimension) {
      auto max_velocity = m_bounds[dimension].first;
      {
        auto temp_min = m_bounds[dimension].first;
        auto temp_max = m_bounds[dimension].second;
        if (temp_min < 0) {
          if (temp_max >= 0) {
            auto temp_min_abs = abs(temp_min);
            if (temp_max > temp_min) {
              max_velocity = temp_min_abs;
            } else {
              max_velocity = temp_max;
            }
          } else {
            max_velocity = abs(temp_max);
          }
        }
      }
      if (new_velocity[dimension] > (max_velocity / 2)) {
        new_velocity[dimension] = max_velocity;
      }
    }
    new_particle.v = new_velocity;
    new_particle.x += new_velocity;

    if (m_compare(m_function(new_particle.x), m_function(new_particle.best))) {
      new_particle.best = new_particle.x;
    }
    return new_particle;
  }

  void update_gbest() {
    for (auto it = std::begin(m_particles); it != std::end(m_particles); ++it) {
      if (m_compare(m_function((*it).best), m_function(m_gbest))) {
        m_gbest = (*it).best;
      }
    }
  }
  container_t<value_t> m_gbest;
};
}