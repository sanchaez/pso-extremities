#pragma once
#include <initializer_list>
#include <iomanip>
#include <iostream>

template <class T>
inline void fixed_width_print_obj(T obj, const int& width = 20,
                                  const int& precision = 10)
{
  std::cout << std::left << std::setw(width) << std::setprecision(precision)
            << obj;
}

template <class T>
void fixed_width_print_line(std::initializer_list<T> list,
                            const int&               width = 20)
{
  for (T obj : list)
  {
    fixed_width_print_obj<T>(obj, width);
  }
  std::cout << std::endl;
}

template <typename T>
void fixed_width_print_line(T list, const int& width = 20)
{
  typedef typename T::value_type value_type;
  for (value_type obj : list)
  {
    fixed_width_print_obj<value_type>(obj, width);
  }
  std::cout << std::endl;
}

template <typename T>
void fixed_width_print_matrix(T matrix, const int& width = 20)
{
  typedef typename T::value_type value_type_line;
  for (value_type_line line : matrix)
  {
    fixed_width_print_line<value_type_line>(line, width);
  }
  std::cout << std::endl;
}