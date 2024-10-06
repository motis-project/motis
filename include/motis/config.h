#pragma once

#include <filesystem>
#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include "utl/verify.h"

namespace motis {

using headers_t = std::map<std::string, std::string>;

enum class feature {
  GEOCODING,
  REVERSE_GEOCODING,
  TIMETABLE,
  STREET_ROUTING,
  OSR_FOOTPATH,
  ELEVATORS,
  TILES
};

struct config {
  friend std::ostream& operator<<(std::ostream&, config const&);
  static config read(std::filesystem::path const&);
  static config read(std::string const&);

  bool has_feature(feature) const;
  void verify() const;

  bool operator==(config const&) const = default;

  std::optional<std::set<feature>> features_{};

  struct server {
    bool operator==(server const&) const = default;
    std::string host_{"0.0.0.0"};
    std::string port_{"8080"};
    unsigned n_threads_{std::thread::hardware_concurrency()};
  };
  std::optional<server> server_{};

  std::optional<std::filesystem::path> osm_{};
  std::optional<std::filesystem::path> fasta_{};

  struct tiles {
    bool operator==(tiles const&) const = default;
    std::filesystem::path profile_;
    std::optional<std::filesystem::path> coastline_;
    std::size_t db_size_{sizeof(void*) >= 8
                             ? 1024ULL * 1024ULL * 1024ULL * 1024ULL
                             : 256U * 1024U * 1024U};
    std::size_t flush_threshold_{sizeof(void*) >= 8 ? 10'000'000 : 100'000};
  };
  std::optional<tiles> tiles_{};

  struct timetable {
    struct dataset {
      struct rt {
        bool operator==(rt const&) const = default;
        std::string url_;
        std::optional<headers_t> headers_{};
      };

      bool operator==(dataset const&) const = default;

      std::filesystem::path path_;
      bool default_bikes_allowed_{false};
      std::optional<std::map<std::string, bool>> clasz_bikes_allowed_{};
      std::optional<std::vector<rt>> rt_{};
      std::optional<std::string> default_timezone_{};
    };

    bool operator==(timetable const&) const = default;
    std::string first_day_{"TODAY"};
    std::uint16_t num_days_{365U};
    bool with_shapes_{true};
    bool ignore_errors_{false};
    bool adjust_footpaths_{true};
    bool merge_dupes_intra_src_{false};
    bool merge_dupes_inter_src_{false};
    unsigned link_stop_distance_{100U};
    std::uint16_t max_footpath_length_{15};
    std::optional<std::string> default_timezone_{};
    std::map<std::string, dataset> datasets_{};
    std::optional<std::filesystem::path> assistance_times_{};
  };
  std::optional<timetable> timetable_{};
};

}  // namespace motis