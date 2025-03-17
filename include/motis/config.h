#pragma once

#include <filesystem>
#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include "cista/hashing.h"

#include "utl/verify.h"

namespace motis {

using headers_t = std::map<std::string, std::string>;

struct config {
  friend std::ostream& operator<<(std::ostream&, config const&);
  static config read_simple(std::vector<std::string> const& args);
  static config read_legacy(std::filesystem::path const&);
  static config read(std::filesystem::path const&);
  static config read(std::string const&);

  void verify() const;
  void verify_input_files_exist() const;

  bool requires_rt_timetable_updates() const;
  bool has_gbfs_feeds() const;
  bool has_odm() const;

  bool operator==(config const&) const = default;

  struct server {
    bool operator==(server const&) const = default;
    std::string host_{"0.0.0.0"};
    std::string port_{"8080"};
    std::string web_folder_{"ui"};
    unsigned n_threads_{std::thread::hardware_concurrency()};
  };
  std::optional<server> server_{};

  std::optional<std::filesystem::path> osm_{};
  std::optional<std::filesystem::path> fasta_{};

  struct tiles {
    bool operator==(tiles const&) const = default;
    std::filesystem::path profile_;
    std::optional<std::filesystem::path> coastline_{};
    std::size_t db_size_{sizeof(void*) >= 8
                             ? 256ULL * 1024ULL * 1024ULL * 1024ULL
                             : 256U * 1024U * 1024U};
    std::size_t flush_threshold_{100'000};
  };
  std::optional<tiles> tiles_{};

  struct timetable {
    struct dataset {
      struct rt {
        bool operator==(rt const&) const = default;
        cista::hash_t hash() const noexcept {
          return cista::build_hash(url_, headers_);
        }
        std::string url_;
        std::optional<headers_t> headers_{};
      };

      bool operator==(dataset const&) const = default;

      std::string path_;
      bool default_bikes_allowed_{false};
      std::optional<std::map<std::string, bool>> clasz_bikes_allowed_{};
      std::optional<std::vector<rt>> rt_{};
      std::optional<std::string> default_timezone_{};
    };

    bool operator==(timetable const&) const = default;
    std::string first_day_{"TODAY"};
    std::uint16_t num_days_{365U};
    bool railviz_{true};
    bool with_shapes_{true};
    bool adjust_footpaths_{true};
    bool merge_dupes_intra_src_{false};
    bool merge_dupes_inter_src_{false};
    unsigned link_stop_distance_{100U};
    unsigned update_interval_{60};
    unsigned http_timeout_{30};
    bool incremental_rt_update_{false};
    bool use_osm_stop_coordinates_{false};
    bool extend_missing_footpaths_{false};
    std::uint16_t max_footpath_length_{15};
    double max_matching_distance_{25.0};
    std::optional<std::string> default_timezone_{};
    std::map<std::string, dataset> datasets_{};
    std::optional<std::filesystem::path> assistance_times_{};
  };
  std::optional<timetable> timetable_{};

  struct gbfs {
    bool operator==(gbfs const&) const = default;

    struct restrictions {
      bool operator==(restrictions const&) const = default;
      bool ride_start_allowed_{true};
      bool ride_end_allowed_{true};
      bool ride_through_allowed_{true};
    };

    struct feed {
      bool operator==(feed const&) const = default;
      std::string url_;
      std::optional<headers_t> headers_{};
    };

    std::map<std::string, feed> feeds_{};
    std::map<std::string, restrictions> default_restrictions_{};
    unsigned update_interval_{60};
    unsigned http_timeout_{30};
    unsigned cache_size_{50};
    std::optional<std::string> proxy_{};
  };
  std::optional<gbfs> gbfs_{};

  struct odm {
    bool operator==(odm const&) const = default;
    std::string url_{};
    std::optional<std::string> bounds_{};
  };
  std::optional<odm> odm_{};

  bool street_routing_{false};
  bool osr_footpath_{false};
  bool elevators_{false};
  bool geocoding_{false};
  bool reverse_geocoding_{false};
};

}  // namespace motis
