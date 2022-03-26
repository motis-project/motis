#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "utl/erase_if.h"

namespace motis::csa {

template <typename T, typename Domination>
struct pareto_set {
  template <typename Fn, typename On = std::enable_if_t<!std::is_same_v<T, Fn>>>
  explicit pareto_set(Fn&& fn) : dominates_{std::forward<Domination>(fn)} {}

  pareto_set(pareto_set const&) = delete;
  pareto_set(pareto_set&&) noexcept = default;
  pareto_set& operator=(pareto_set const&) = delete;
  pareto_set& operator=(pareto_set&&) noexcept = default;
  ~pareto_set() = default;

  typename std::vector<T>::const_iterator begin() const { return set_.begin(); }
  typename std::vector<T>::const_iterator end() const { return set_.end(); }
  typename std::vector<T>::iterator begin() { return set_.begin(); }
  typename std::vector<T>::iterator end() { return set_.end(); }

  friend typename std::vector<T>::iterator begin(pareto_set& p) {
    return p.set_.begin();
  }
  friend typename std::vector<T>::iterator end(pareto_set& p) {
    return p.set_.end();
  }
  friend typename std::vector<T>::const_iterator begin(pareto_set const& p) {
    return p.set_.begin();
  }
  friend typename std::vector<T>::const_iterator end(pareto_set const& p) {
    return p.set_.end();
  }

  template <typename Arg>
  bool push_back(Arg&& candidate) {
    if (std::any_of(std::begin(set_), std::end(set_),
                    [&](T const& existing_entry) {
                      return dominates_(existing_entry, candidate);
                    })) {
      return false;
    }
    utl::erase_if(set_, [&](auto const& existing_entry) {
      return dominates_(candidate, existing_entry);
    });
    set_.emplace_back(std::forward<Arg>(candidate));
    return true;
  }

  size_t size() const { return set_.size(); }

  std::vector<T> set_;
  Domination dominates_;
};

template <typename J, typename Fn>
inline pareto_set<J, Fn> make_pareto_set(Fn&& fn) {
  return pareto_set<J, Fn>(std::forward<Fn>(fn));
}

}  // namespace motis::csa
