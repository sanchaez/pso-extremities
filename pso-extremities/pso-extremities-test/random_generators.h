#pragma once
// Random generators for PSO algorithms
#include <random>
#include "pso-types.h"

namespace pso
{
template <typename T>
class AbstractGenerator
{
public:
    AbstractGenerator(const T left_bound, const T right_bound, const int seed = 0)
        : m_left_bound(left_bound), m_right_bound(right_bound), m_seed(seed)
    {
    };
    virtual container_t<T> random_vector(const int size) = 0;
    virtual T random() = 0;
    T operator()()
    {
        return random();
    }
protected:
    const T m_left_bound;
    const T m_right_bound;
    const int m_seed;
};

template <typename T,
    class distribution_t = std::uniform_real_distribution<T>,
    class generator_t = std::default_random_engine>
    class StdGenerator : public AbstractGenerator<T>
{
public:
    StdGenerator(const T left_bound, const T right_bound, const int seed = 0)
        : AbstractGenerator(left_bound, right_bound, seed),
        m_distribution(left_bound, right_bound), m_generator(seed)
    {
    }
    container_t<T> random_vector(const int size)
    {
        container_t<T> vector(T(0), size);
#pragma omp parallel for
        for (int i = 0; i<size; ++i)
        {
            vector[i] = m_distribution(m_generator);
        }
        return vector;
    }
    T random()
    {
        return m_distribution(m_generator);
    }

private:
    distribution_t m_distribution;
    generator_t m_generator;
};
}  // namespace pso
