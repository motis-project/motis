#pragma once

namespace motis::routing {

template <typename... Traits>
struct initializer;

template <typename FirstInitializer, typename... RestInitializer>
struct initializer<FirstInitializer, RestInitializer...> {
  template <typename Label, typename LowerBounds>
  static void init(Label& l, LowerBounds& lb) {
    FirstInitializer::init(l, lb);
    initializer<RestInitializer...>::init(l, lb);
  }
};

template <>
struct initializer<> {
  template <typename Label, typename LowerBounds>
  static void init(Label&, LowerBounds&) {}
};

}  // namespace motis::routing
