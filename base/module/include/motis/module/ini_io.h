#pragma once

#include <map>

#include "boost/filesystem/path.hpp"

#include "boost/property_tree/ini_parser.hpp"
#include "boost/property_tree/ptree.hpp"

#include "cista/decay.h"
#include "cista/hash.h"
#include "cista/reflection/for_each_field.h"

#include "utl/const_str.h"
#include "utl/read_file.h"

namespace motis::module {

template <char const* Str>
struct named_name {
  static constexpr auto const name = Str;
};
#define MOTIS_NAME(str) motis::module::named_name<STRING_LITERAL(str)>

template <typename T, typename... Tags>
struct named : Tags... {
  using value_type = T;
  named() = default;
  named(T param) : t_(param) {}  // NOLINT
  operator T() { return t_; }  // NOLINT
  bool operator==(named const& o) const { return val() == o.val(); }
  T const& val() const { return t_; }
  T& val() { return t_; }
  T t_{};
};

template <typename T>
constexpr auto get_name(T) {
  return T::name;
}

template <typename T>
void write_ini(std::ostream& out, T const& s) {
  cista::for_each_field(s, [&](auto&& field) {
    out << get_name(field) << "=" << field.val() << "\n";
  });
}

template <typename T>
void write_ini(boost::filesystem::path const& p, T const& s) {
  std::ofstream f;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  f.open(p.generic_string().c_str());
  write_ini(f, s);
}

template <typename T>
T read_ini(std::string const& s) {
  namespace pt = boost::property_tree;

  std::stringstream ss;
  ss.str(s);

  pt::ptree tree;
  pt::read_ini(ss, tree);

  T el;
  cista::for_each_field(el, [&](auto&& f) {
    using ValueType = cista::decay_t<decltype(f.val())>;
    f.val() = tree.template get<ValueType>(pt::path{get_name(f)}, ValueType{});
  });
  return el;
}

template <typename T>
T read_ini(boost::filesystem::path const& p) {
  auto const file_content = utl::read_file(p.generic_string().c_str());
  return file_content.has_value() ? read_ini<T>(*file_content) : T{};
}

}  // namespace motis::module