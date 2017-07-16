#pragma once
// Random generators for PSO algorithms
#include "pso-types.h"

#include <random>
#include <algorithm>

namespace pso
{
template <typename T, typename RNGClass>
class RandomGenerator
{
public:
    void reset()
    {
        return static_cast<RNGClass*>(this)->__reset();
    }

    void random_vector(const int size, container_t<T>& vector)
    {
        return static_cast<RNGClass*>(this)->__impl_random_vector(size, vector);
    };

    container_t<T> random_vector_new(const int size)
    {
        return static_cast<RNGClass*>(this)->__impl_random_vector_new(size);
    };

    T random()
    {
        return static_cast<RNGClass*>(this)->__impl_random();
    }

    T operator()()
    {
        return random();
    }
};

template <typename T,
    class generator_t = std::mt19937_64>
    class StdGenerator : public RandomGenerator<T, StdGenerator<T, generator_t>>
{
    friend RandomGenerator;

    void __reset()
    {
        m_generator.seed(__impl_random());
        m_distribution.reset();
    }

    container_t<T> __impl_random_vector_new(const int size)
    {
        container_t<T> vector(size);
        __impl_random_vector_noresize(vector);
        return vector;
    }

    void __impl_random_vector(const int size, container_t<T>& vector)
    {
        vector.resize(size);
        __impl_random_vector_noresize(vector);
    }

    inline void __impl_random_vector_noresize(container_t<T>& vector)
    {
        std::generate(std::begin(vector), std::end(vector), [&]()
        {
            return __impl_random();
        });
    }

    inline T __impl_random()
    {
        return m_distribution(m_generator);
    }

public:
    StdGenerator(const T left_bound, const T right_bound, const int seed = rand())
        : m_distribution(left_bound, right_bound), m_generator(seed)
    {
    }

private:
    std::uniform_real_distribution<T> m_distribution;
    generator_t m_generator;
};
}  // namespace pso
