#pragma once

namespace motis::routing {

template <typename... Traits>
struct filter;

template <typename FirstFilter, typename... RestFilters>
struct filter<FirstFilter, RestFilters...> {
  template <typename Label>
  static bool is_filtered(Label const& l) {
    return FirstFilter::is_filtered(l) ||
           filter<RestFilters...>::is_filtered(l);
  }
};

template <>
struct filter<> {
  template <typename Label>
  static bool is_filtered(Label const&) {
    return false;
  }
};

}  // namespace motis::routing
