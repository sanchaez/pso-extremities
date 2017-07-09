/* PSO interface */

#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <valarray>
#include "pso-types.h"
#include "random_generators.h"

namespace pso
{
template <typename value_t>
dimension_container_t<value_t> unified_bounds(
    const dimension_limit_t<value_t>& limit,
    const int size)
{
    return dimension_container_t<value_t>(limit, size);
}
template <typename value_t>
dimension_container_t<value_t> unified_bounds(const value_t left_bound,
                                              const value_t right_bound,
                                              const int size)
{
    return dimension_container_t<value_t>(std::make_pair(left_bound, right_bound),
                                          size);
}

template <typename value_t>
struct Particle
{
    Particle() : v(0), x(0), best(0), best_ever(0) {}
    Particle(const container_t<value_t>& coordinates, const value_t& velocity)
        : x(coordinates),
        best(coordinates),
        best_ever(coordinates),
        v(velocity)
    {
    }
    container_t<value_t> v;
    container_t<value_t> x;
    container_t<value_t> best;
    container_t<value_t> best_ever;
};

template <typename value_t>
using particle_container_t = container_t<Particle<value_t>>;
/// Abstract class for all particle swarms

template <typename value_t>
class BasePSO
{
public:
    BasePSO(const predicate_t<value_t>& predicate,
            const int particles_number,
            const dimension_container_t<value_t>& boundaries,
            const function_t<value_t>& function)
        : m_bounds(boundaries),
        m_compare(predicate),
        m_dimensions_number(boundaries.size()),
        m_particles_number(particles_number),
        m_particles(particles_number),
        m_function(function)
    {
    }

protected:
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

    int m_particles_number;
    int m_dimensions_number;
    function_t<value_t> m_function;
    predicate_t<value_t> m_compare;
    dimension_container_t<value_t> m_bounds;
    particle_container_t<value_t> m_particles;
};

template <typename value_t, class generator_t = StdGenerator<value_t>>
class AbstractPSO : public BasePSO<value_t>
{
public:
    using BasePSO<value_t>::BasePSO;
    virtual void initialize_particles()
    {
        m_particles.resize(m_particles_number);
        // init every dimension separately
#pragma omp parallel
        {
#pragma omp for
            for (int particle = 0; particle < m_particles_number; ++particle)
            {
                m_particles[particle].x.resize(m_dimensions_number);
                m_particles[particle].v.resize(m_dimensions_number);
                m_particles[particle].best.resize(m_dimensions_number);
            }

            for (int dimension = 0; dimension < m_dimensions_number; ++dimension)
            {
                generator_t dimension_distribution(m_bounds[dimension].first,
                                                   m_bounds[dimension].second);
                generator_t velocity_distribution(0, m_bounds[dimension].second);
#pragma omp for schedule(dynamic)
                for (int particle = 0; particle < m_particles_number; ++particle)
                {
                    m_particles[particle].x[dimension] = dimension_distribution();
                    m_particles[particle].v[dimension] = velocity_distribution();
                    m_particles[particle].best[dimension] =
                        m_particles[particle].x[dimension];
                }
            }
        }
    }
    virtual value_coordinates_t<value_t> operator()(const int iterations_max) = 0;
};

template <typename value_t, class generator_t = StdGenerator<value_t>>
class ClassicGbestPSO : public AbstractPSO<value_t, generator_t>
{
public:
    ClassicGbestPSO(const predicate_t<value_t>& predicate,
                    const int particles_number,
                    const dimension_container_t<value_t>& boundaries,
                    const function_t<value_t>& function)
        : AbstractPSO<value_t>(predicate, particles_number, boundaries, function),

        m_eps_generator(0, 1)
    {
    }

    value_coordinates_t<value_t> operator()(const int iterations_max)
    {
        initialize_particles();
#pragma omp parallel
        {
            for (int i = 0; i < iterations_max; ++i)
            {
#pragma omp for
                for (int j = 0; j < m_particles_number; ++j)
                {
                    m_particles[j] = update_particle(m_particles[j]);
                }
#pragma omp single
                update_best_ever();
            }
        }
        return std::make_pair(m_function(m_gbest), m_gbest);
    }


    void initialize_particles()
    {
        AbstractPSO::initialize_particles();
        // init gbest
        m_gbest = m_particles[0].x;
        update_best_ever();
    }

private:
    Particle<value_t> update_particle(const Particle<value_t>& p)
    {
        Particle<value_t> new_particle;
        container_t<value_t> eps1(m_eps_generator.random_vector(m_dimensions_number));
        container_t<value_t> eps2(m_eps_generator.random_vector(m_dimensions_number));


        new_particle.v = value_t(0.72984) * (p.v + (p.best - p.x) * eps1 * 2.05 +
            (m_gbest - p.x) * eps2 * 2.05);
        new_particle.x = new_particle.v + p.x;
        if (compare_function_values(new_particle.x, p.best))
        {
            new_particle.best = new_particle.x;
        }
        else
        {
            new_particle.best = p.best;
        }

        return new_particle;
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

    container_t<value_t> m_gbest;

    container_t<value_t> eps1;
    container_t<value_t> eps2;
    generator_t m_eps_generator;
};
}  // namespace pso