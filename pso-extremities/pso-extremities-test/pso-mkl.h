#pragma once
#include <mkl_vsl.h>
#include "pso.h"
#define DEFAULT_GENERATOR VSL_BRNG_SFMT19937
#define SEED 1

namespace pso {
class AbstractMKLPSO : public BasePSO<double> {
 public:
  using BasePSO::BasePSO;

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
        update_best_ever();
      }
    }

    vslDeleteStream(&random_stream_eps1);
    vslDeleteStream(&random_stream_eps2);
    delete eps1_v;
    delete eps2_v;
    return result();
  }

 protected:
  virtual void update_best_ever() = 0;
  virtual value_coordinates_t<double> result() = 0;

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
        m_particles[particle].best_ever.resize(m_dimensions_number);
      }
      for (int particle = 0; particle < m_particles_number; ++particle) {
#pragma omp for schedule(guided)
        for (int dimension = 0; dimension < m_dimensions_number; ++dimension) {
          vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream_x, 1,
                       &uniform_random_number_x, m_bounds[dimension].first,
                       m_bounds[dimension].second);
          vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream_v, 1,
                       &uniform_random_number_v, m_bounds[dimension].first,
                       m_bounds[dimension].second);
          m_particles[particle].x[dimension] = uniform_random_number_x;
          m_particles[particle].v[dimension] = uniform_random_number_v;
        }
        m_particles[particle].best = m_particles[particle].x;
        m_particles[particle].best_ever = m_particles[particle].x;
      }
    }
    vslDeleteStream(&random_stream_v);
    vslDeleteStream(&random_stream_x);
    //
    update_best_ever();
  }

  Particle<double> __vectorcall update_particle(const Particle<double>& p) {
    Particle<double> new_particle = p;
    container_t<double> eps1;
    container_t<double> eps2;

#pragma omp parallel sections
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

    new_particle.v = double(0.72984) * (p.v + (p.best - p.x) * eps1 * 2.05 +
                                        (p.best_ever - p.x) * eps2 * 2.05);
    new_particle.x = new_particle.v + p.x;

    if (!in_bounds(new_particle.x)) {
      bounds_return(new_particle.x);
      new_particle.v = new_particle.x - p.x;
    }

    if (compare_function_values(new_particle.x, p.best)) {
      new_particle.best = new_particle.x;
    } else {
      new_particle.best = p.best;
    }
    return new_particle;
  }

  double* eps1_v;
  double* eps2_v;
  VSLStreamStatePtr random_stream_eps1;
  VSLStreamStatePtr random_stream_eps2;
};

class ClassicGbestMKLPSO : public AbstractMKLPSO {
 public:
  using AbstractMKLPSO::AbstractMKLPSO;

 protected:
  void update_best_ever() {
    container_t<double> m_gbest = m_particles[0].best_ever;
#pragma omp parallel
    {
#pragma omp for
      // determine best
      for (int i = 0; i < m_particles_number; ++i) {
        if (compare_function_values(m_particles[i].best, m_gbest)) {
#pragma omp critical
          m_gbest = m_particles[i].best;
        }
      }
#pragma omp for
      // write to particles
      for (int i = 0; i < m_particles_number; ++i) {
        m_particles[i].best_ever = m_gbest;
      }
    }
  }
  virtual value_coordinates_t<double> result() {
    return std::make_pair(m_function(m_particles[0].best_ever),
                          m_particles[0].best_ever);
  }
};

class ClassicLbestMKLPSO : public AbstractMKLPSO {
  using AbstractMKLPSO::AbstractMKLPSO;

  void __vectorcall compare_assign_local(pso::container_t<double>& m,
                            const pso::container_t<double>& l,
                            const pso::container_t<double>& r) {
    if (compare_function_values(l, m)) {
      m = l;
    }
    if (compare_function_values(r, m)) {
      m = r;
    }
  }

  void update_best_ever() {
    compare_assign_local(m_particles[0].best_ever,
                         m_particles[m_particles_number - 1].best,
                         m_particles[1].best);

    for (int i = 1; i < m_particles_number - 1; ++i) {
      compare_assign_local(m_particles[i].best_ever, m_particles[i - 1].best,
                           m_particles[i + 1].best);
    }
    compare_assign_local(m_particles[m_particles_number - 1].best_ever,
                         m_particles[m_particles_number - 2].best,
                         m_particles[0].best);
  }

  value_coordinates_t<double> result() {
    container_t<double> best_ever = m_particles[0].best_ever;

#pragma omp parallel for
      // determine best
      for (int i = 0; i < m_particles_number; ++i) {
        if (compare_function_values(m_particles[i].best_ever, best_ever)) {
#pragma omp critical
          best_ever = m_particles[i].best_ever;
        }
      }
    
    return std::make_pair(m_function(best_ever), best_ever);
  }
};
}