#pragma once

#include <cinttypes>
#include <cmath>
#include <algorithm>
#include <bitset>
#include <map>
#include <memory>
#include <string>

#include "boost/filesystem.hpp"

#include "flatbuffers/flatbuffers.h"

#include "utl/to_vec.h"

#include "utl/parser/buffer.h"
#include "utl/parser/cstr.h"
#include "utl/parser/file.h"

namespace motis::loader {

std::string pad_to_7_digits(int eva_num);

template <typename T>
inline flatbuffers64::Offset<flatbuffers64::String> to_fbs_string(
    flatbuffers64::FlatBufferBuilder& b, T const& s) {
  return b.CreateString(s.c_str(), s.length());
}

template <typename T>
flatbuffers64::Offset<flatbuffers64::String> to_fbs_string(
    flatbuffers64::FlatBufferBuilder& b, T const& s,
    std::string const& /* charset -> currently only supported: ISO-8859-1 */) {
  std::vector<unsigned char> utf8(s.length() * 2, '\0');
  auto number_of_input_bytes = s.length();
  auto in = reinterpret_cast<unsigned char const*>(s.c_str());
  auto out_begin = &utf8[0];
  auto out = out_begin;
  for (std::size_t i = 0; i < number_of_input_bytes; ++i) {
    if (*in < 128) {
      *out++ = *in++;
    } else {
      *out++ = 0xc2 + (*in > 0xbfU);
      *out++ = (*in++ & 0x3fU) + 0x80U;
    }
  }
  return to_fbs_string(b, utl::cstr(reinterpret_cast<char const*>(out_begin),
                                    std::distance(out_begin, out)));
}

template <typename MapType>
inline std::vector<typename MapType::value_type::second_type> values(
    MapType const& m) {
  return utl::to_vec(m, [](auto&& el) { return el.second; });
}

template <typename IntType>
inline IntType raw_to_int(utl::cstr s) {
  IntType key = 0;
  std::memcpy(&key, s.str, std::min(s.len, sizeof(IntType)));
  return key;
}

template <typename It, typename Predicate>
inline It find_nth(It begin, It end, std::size_t n, Predicate fun) {
  assert(n != 0);
  std::size_t num_elements_found = 0;
  auto it = begin;
  while (it != end && num_elements_found != n) {
    it = std::find_if(it, end, fun);
    ++num_elements_found;
    if (it != end && num_elements_found != n) {
      ++it;
    }
  }
  return it;
}

template <typename TargetCollection, typename It, typename UnaryOperation>
inline TargetCollection transform(It s, It e, UnaryOperation op) {
  TargetCollection c;
  std::transform(s, e, std::back_insert_iterator<TargetCollection>(c), op);
  return c;
}

template <typename T>
inline std::vector<T> repeat_n(T const& el, std::size_t n) {
  std::vector<T> els(n);
  std::fill(begin(els), end(els), el);
  return els;
}

inline int yyyymmdd_year(int yyyymmdd) { return yyyymmdd / 10000; }
inline int yyyymmdd_month(int yyyymmdd) { return (yyyymmdd % 10000) / 100; }
inline int yyyymmdd_day(int yyyymmdd) { return yyyymmdd % 100; }

void write_schedule(flatbuffers64::FlatBufferBuilder& b,
                    boost::filesystem::path const& path);

std::size_t collect_files(boost::filesystem::path const& root,
                          std::string const& file_extension,
                          std::vector<boost::filesystem::path>& files);

}  // namespace motis::loader
