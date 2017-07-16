/* PSO interface */

#pragma once

#include "pso-types.h"
#include "random_generators.h"

#include <algorithm>

namespace pso
{

dimension_container_t unified_bounds(const dimension_limit_t& limit,
                                              const int size)
{
    return dimension_container_t(size, limit);
}

dimension_container_t unified_bounds(const double left_bound,
                                              const double right_bound,
                                              const int size)
{
    return dimension_container_t(size, std::make_pair(left_bound, right_bound));
}

struct Particle
{
    Particle() {}
    Particle(const container_t<value_t>& coordinates, const value_t& velocity)
        : x(coordinates),
        best(coordinates),
        //best_ever(coordinates),
        v(coordinates.size(), velocity)
    {
    }
    container_t<value_t> v;
    container_t<value_t> x;
    container_t<value_t> best;
};

using particle_container_t = container_t<Particle>;

/// Abstract class for all particle swarms
template <typename PSOClass>
class AbstractPSO
{
public:
    AbstractPSO(const predicate_t& predicate,
                const int particles_number,
                const dimension_container_t& boundaries,
                const function_t& function)
        : m_bounds(boundaries),
        m_compare(predicate),
        m_dimensions_number(boundaries.size()),
        m_particles_number(particles_number),
        m_particles(particles_number),
        m_function(function)
    {
    }

    value_coordinates_t operator()(const int iterations_max)
    {
        return static_cast<PSOClass*>(this)->__impl_function(iterations_max);
    };

protected:
    // used to call compare function
    
    inline constexpr bool compare_function_values(
        const container_t<value_t>& a,
        const container_t<value_t>& b) const
    {
        return m_compare(m_function(a), m_function(b));
    }

    bool in_bounds(container_t<value_t>& point_coordinates)
    {
        for (int i = 0; i < m_dimensions_number; ++i)
        {
            if (m_bounds[i].first > point_coordinates[i] ||
                m_bounds[i].second < point_coordinates[i])
            {
                return false;
            }
        }
        return true;
    }

    // checks if point is in bounds and retuns it to the edge
    void bounds_return(container_t<value_t>& point_coordinates)
    {
        for (int i = 0; i < m_dimensions_number; ++i)
        {
            if (m_bounds[i].first > point_coordinates[i])
            {
                point_coordinates[i] = m_bounds[i].first;
            }
            else if (m_bounds[i].second < point_coordinates[i])
            {
                point_coordinates[i] = m_bounds[i].second;
            }
        }
    }

    void initialize_particles()
    {
        m_particles.resize(m_particles_number);
#pragma omp parallel
        {
            for (int dimension = 0; dimension < m_dimensions_number; ++dimension)
            {
                StdGenerator<value_t> dimension_distribution(m_bounds[dimension].first,
                                                   m_bounds[dimension].second);
                StdGenerator<value_t> velocity_distribution(0, m_bounds[dimension].second);
#pragma omp for
                for (int particle = 0; particle < m_particles_number; ++particle)
                {
                    dimension_distribution.random_vector(m_dimensions_number, m_particles[particle].x);
                    velocity_distribution.random_vector(m_dimensions_number, m_particles[particle].v);
                    m_particles[particle].best = m_particles[particle].x;
                }
            }
        }
    }

    int m_particles_number;
    int m_dimensions_number;

    function_t m_function;
    predicate_t m_compare;
    dimension_container_t m_bounds;
    particle_container_t m_particles;
};


template <typename generator_t = StdGenerator<value_t>, int c1 = 2.05, int c2 = 2.05>
class GbestPSO : public AbstractPSO<GbestPSO<generator_t, c1, c2>>
{
public:
    using BasePSO = AbstractPSO<GbestPSO<generator_t, c1, c2>>;
    GbestPSO(const predicate_t& predicate,
             const int particles_number,
             const dimension_container_t& boundaries,
             const function_t& function)
        : BasePSO(predicate, particles_number, boundaries, function),
        m_eps_gen(0, 1)
    {
    }

private:
    friend BasePSO;

    value_coordinates_t __impl_function(const int iterations_max)
    {
        initialize_particles();
#pragma omp parallel
        {
            for (int i = 0; i < iterations_max; ++i)
            {
#pragma omp for
                for (int j = 0; j < m_particles_number; ++j)
                {
                    update_particle(m_particles[j], m_eps_gen.random_vector_new(2 * m_dimensions_number));
                }

#pragma omp single
                {
                    update_best_ever();
                    m_eps_gen.reset();
                }
            }
        }
        return std::make_pair(m_function(m_gbest), m_gbest);
    }

protected:
    inline void initialize_particles()
    {
        BasePSO::initialize_particles();
        init_gbest();
    }

    inline void init_gbest()
    {
        m_gbest = m_particles[0].x;
        update_best_ever();
    }


    inline void update_particle(Particle& p,
                                const container_t<value_t>& eps)
    {
#pragma omp parallel for
        for (int i = 0; i < m_dimensions_number; ++i)
        {
            p.v[i] += (p.best[i] - p.x[i]) * eps[i] * c1
                + (m_gbest[i] - p.x[i]) * eps[m_dimensions_number + i] * c2;
            p.v[i] *= value_t(0.72984);
            p.x[i] += p.v[i];
        }

        if (compare_function_values(p.x, p.best))
        {
            p.best = p.x;
        }
    }

    void update_best_ever()
    {
#pragma omp parallel for
        for (int i = 0; i < m_particles_number; ++i)
        {
            if (compare_function_values(m_particles[i].best, m_gbest))
            {
#pragma omp critical
                { m_gbest = m_particles[i].best; }
            }
        }
    }
    generator_t m_eps_gen;
    container_t<value_t> m_gbest;
};

using ClassicGbestPSO = GbestPSO<StdGenerator<double>>;

}  // namespace pso

