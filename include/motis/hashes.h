#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <utility>

namespace motis {

using meta_entry_t = std::pair<std::string, std::uint64_t>;
using meta_t = std::map<std::string, std::uint64_t>;

constexpr auto const osr_version = []() {
  return meta_entry_t{"osr_bin_ver", 11U};
};
constexpr auto const adr_version = []() {
  return meta_entry_t{"adr_bin_ver", 4U};
};
constexpr auto const n_version = []() {
  return meta_entry_t{"nigiri_bin_ver", 9U};
};
constexpr auto const matches_version = []() {
  return meta_entry_t{"matches_bin_ver", 4U};
};
constexpr auto const tiles_version = []() {
  return meta_entry_t{"tiles_bin_ver", 1U};
};
constexpr auto const osr_footpath_version = []() {
  return meta_entry_t{"osr_footpath_bin_ver", 1U};
};

std::string to_str(meta_t const&);

meta_t read_hashes(std::filesystem::path const& data_path,
                   std::string const& name);

void write_hashes(std::filesystem::path const& data_path,
                  std::string const& name,
                  meta_t const& h);

}  // namespace motis