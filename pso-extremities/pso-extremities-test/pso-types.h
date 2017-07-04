#pragma once
// Types for PSO

namespace pso
{
template <typename value_t>
using container_t = std::valarray<value_t>;
template <typename value_t>
using dimension_limit_t = std::pair<value_t, value_t>;
template <typename value_t>
using dimention_container_t = container_t<dimension_limit_t<value_t>>;
template <typename value_t>
using predicate_t = std::function<bool(value_t, value_t)>;
template <typename value_t>
using stop_predicate_t = std::function<bool(value_t)>;
template <typename value_t>
using function_t = std::function<value_t(container_t<value_t>)>;
template <typename value_t>
using value_coordinates_t = std::pair<value_t, container_t<value_t>>;


}