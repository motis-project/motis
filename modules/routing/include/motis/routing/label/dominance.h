#pragma once

namespace motis::routing {

template <typename TieBreaker, typename... Dominators>
struct dominance;

template <typename TieBreaker, typename FirstDominator,
          typename... RestDominators>
struct dominance<TieBreaker, FirstDominator, RestDominators...> {
  template <typename Label>
  static bool dominates(bool could_dominate, Label const& a, Label const& b,
                        bool terminal) {
    auto d = FirstDominator::dominates(a, b, terminal);
    return !d.greater() && dominance<TieBreaker, RestDominators...>::dominates(
                               could_dominate || d.smaller(), a, b, terminal);
  }
};

template <typename TieBreaker>
struct dominance<TieBreaker> {
  template <typename Label>
  static bool dominates(bool could_dominate, Label const& a, Label const& b,
                        bool) {
    return TieBreaker::dominates(could_dominate, a, b);
  }
};

}  // namespace motis::routing