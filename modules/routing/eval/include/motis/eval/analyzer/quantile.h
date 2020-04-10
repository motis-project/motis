#ifndef IMOTIS_EVAL_QUANTILE_H_
#define IMOTIS_EVAL_QUANTILE_H_

#include <cmath>
#include <algorithm>
#include <type_traits>

namespace motis::eval {

// From http://stackoverflow.com/a/10642935
template <typename T>
struct member_type_helper;
template <typename R, typename T>
struct member_type_helper<R(T::*)> {
  using type = R;
  using parent_type = T;
};
template <typename T>
struct member_type : public member_type_helper<T> {};

template <typename Attr, typename It>
It quantile_it(Attr attr, It begin, It end, double q) {
  using val_type = typename std::remove_reference<
      typename std::remove_cv<decltype(*begin)>::type>::type;
  std::sort(begin, end, [&attr](val_type const& r1, val_type const& r2) {
    return r1.*attr < r2.*attr;
  });
  if (q == 1.0) {
    return begin + std::distance(begin, end) - 1;
  } else {
    return begin + std::round(q * std::distance(begin, end));
  }
}

template <typename Attr, typename It>
typename member_type<Attr>::type quantile(Attr attr, It begin, It end,
                                          double q) {
  return *(quantile_it(attr, begin, end, q)).*attr;
}

template <typename Attr, typename Col>
typename member_type<Attr>::type quantile(Attr attr, Col col, double q) {
  return quantile(attr, std::begin(col), std::end(col), q);
}

}  // namespace motis::eval

#endif  // IMOTIS_EVAL_QUANTILE_H_