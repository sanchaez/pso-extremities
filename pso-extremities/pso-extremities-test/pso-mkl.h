#pragma once
#include <mkl_vsl.h>
#include <ctime>
#include "pso.h"
#define DEFAULT_GENERATOR VSL_BRNG_SFMT19937
#define SEED 1

namespace pso {
class AbstractMKLPSO : public BasePSO<double> {
 public:
  AbstractMKLPSO(const predicate_t<double>& predicate,
                 const int particles_number,
                 const dimention_container_t<double>& boundaries,
                 const function_t<double>& function)
      : BasePSO(predicate, particles_number, boundaries, function) {}

 protected:
  void initialize_particles() {
    m_particles.resize(m_particles_number);
    double uniform_random_number_x;
    double uniform_random_number_v;
    VSLStreamStatePtr random_stream_v;
    VSLStreamStatePtr random_stream_x;
    vslNewStream(&random_stream_x, DEFAULT_GENERATOR, SEED);
    vslNewStream(&random_stream_v, DEFAULT_GENERATOR, SEED);
    vslSkipAheadStream(random_stream_x, m_particles_number);
    vslSkipAheadStream(random_stream_v, 2 * m_particles_number);

// init every dimension separately
#pragma omp parallel
    {
#pragma omp for
      for (int particle = 0; particle < m_particles_number; ++particle) {
        m_particles[particle].x.resize(m_dimensions_number);
        m_particles[particle].v.resize(m_dimensions_number);
        m_particles[particle].best.resize(m_dimensions_number);
      }

      for (int particle = 0; particle < m_particles_number; ++particle) {
#pragma omp for schedule(dynamic)
        for (int dimension = 0; dimension < m_dimensions_number; ++dimension) {
          vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream_x, 1,
                       &uniform_random_number_x, m_bounds[dimension].first,
                       m_bounds[dimension].second);
          vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream_v, 1,
                       &uniform_random_number_v, m_bounds[dimension].first,
                       m_bounds[dimension].second);
          m_particles[particle].x[dimension] = uniform_random_number_x;
          m_particles[particle].v[dimension] = uniform_random_number_v;
          m_particles[particle].best[dimension] =
              m_particles[particle].x[dimension];
        }
      }
    }
    vslDeleteStream(&random_stream_v);
    vslDeleteStream(&random_stream_x);
  }
};

class ClassicGbestMKLPSO : public AbstractMKLPSO {
 public:
  ClassicGbestMKLPSO(const predicate_t<double>& predicate,
                     const int particles_number,
                     const dimention_container_t<double>& boundaries,
                     const function_t<double>& function)
      : AbstractMKLPSO(predicate, particles_number, boundaries, function) {}

  value_coordinates_t<double> operator()(const int iterations_max) {
    initialize_particles();
    eps1_v = new double[m_dimensions_number];
    eps2_v = new double[m_dimensions_number];
    vslNewStream(&random_stream_eps1, DEFAULT_GENERATOR, SEED);
    vslNewStream(&random_stream_eps2, DEFAULT_GENERATOR, SEED);

    vslSkipAheadStream(random_stream_eps1,
                       2 * m_particles_number + 3 * m_dimensions_number);
    vslSkipAheadStream(random_stream_eps2,
                       2 * m_particles_number + 4 * m_dimensions_number);

#pragma omp parallel
    {
      for (int i = 0; i < iterations_max; ++i) {
#pragma omp for schedule(dynamic)
        for (int j = 0; j < m_particles_number; ++j) {
          m_particles[j] = update_particle(m_particles[j]);
        }
#pragma omp single
        update_gbest();
      }
    }
    vslDeleteStream(&random_stream_eps1);
    vslDeleteStream(&random_stream_eps2);
    delete eps1_v;
    delete eps2_v;
    return std::make_pair(m_function(m_gbest), m_gbest);
  }

 private:
  void initialize_particles() {
    AbstractMKLPSO::initialize_particles();
    // init gbest
    m_gbest = m_particles[0].x;
    update_gbest();
  }
  Particle<double> update_particle(const Particle<double>& p) {
    Particle<double> new_particle;
    container_t<double> eps1;
    container_t<double> eps2;
#pragma omp parallel
    {
#pragma omp sections
      {
#pragma omp section
        {
          vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream_eps1,
                       m_dimensions_number, eps1_v, 0.0, 1.0);
          eps1 = container_t<double>(eps1_v, m_dimensions_number);
        }

#pragma omp section
        {
          vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream_eps2,
                       m_dimensions_number, eps2_v, 0.0, 1.0);
          eps2 = container_t<double>(eps2_v, m_dimensions_number);
        }
      }
    }
    new_particle.v = double(0.72984) * (p.v + (p.best - p.x) * eps1 * 2.05 +
                                        (m_gbest - p.x) * eps2 * 2.05);
    new_particle.x = new_particle.v + p.x;

    if (m_compare(m_function(new_particle.x), m_function(p.best))) {
      new_particle.best = new_particle.x;
    } else {
      new_particle.best = p.best;
    }

    return new_particle;
  }
  void update_gbest() {
#pragma omp parallel for
    for (int i = 0; i < m_particles_number; ++i) {
      if (m_compare(m_function(m_particles[i].best), m_function(m_gbest))) {
#pragma omp critical
        { m_gbest = m_particles[i].best; }
      }
    }
  }
  double* eps1_v;
  double* eps2_v;
  VSLStreamStatePtr random_stream_eps1;
  VSLStreamStatePtr random_stream_eps2;
  container_t<double> m_gbest;
};
}