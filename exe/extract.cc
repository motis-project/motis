#include <filesystem>
#include <iterator>

#include "boost/json.hpp"
#include "boost/program_options.hpp"

#include "fmt/ranges.h"
#include "fmt/std.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/split.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/dir.h"
#include "nigiri/common/interval.h"

#include "motis-api/motis-api.h"
#include "motis/tag_lookup.h"
#include "motis/types.h"

#include "flags.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {

void copy_stop_times(hash_set<std::string> const& trip_ids,
                     hash_set<std::string> const& filter_stop_ids,
                     std::string_view file_content,
                     hash_set<std::string>& stop_ids,
                     std::ostream& out) {
  struct csv_stop_time {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  auto n_lines = 0U;
  auto reader = utl::make_buf_reader(file_content);
  auto line = reader.read_line();
  auto const header_permutation = utl::read_header<csv_stop_time>(line);
  out << line.view() << "\n";
  while ((line = reader.read_line())) {
    auto const row = utl::read_row<csv_stop_time>(header_permutation, line);
    if (trip_ids.contains(row.trip_id_->view()) &&
        (filter_stop_ids.empty() ||
         filter_stop_ids.contains(row.stop_id_->view()))) {
      stop_ids.insert(row.stop_id_->view());
      out << line.view() << "\n";
      ++n_lines;
    }
  }

  fmt::println("  stop_times.txt: lines written: {}", n_lines);
}

void copy_stops(hash_set<std::string>& stop_ids,
                std::string_view file_content,
                std::ostream& out,
                bool const filter_stops) {
  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
  };

  {  // First pass: collect parents.
    auto reader = utl::make_buf_reader(file_content);
    auto line = reader.read_line();
    auto const header_permutation = utl::read_header<csv_stop>(line);
    while ((line = reader.read_line())) {
      auto const row = utl::read_row<csv_stop>(header_permutation, line);
      if (!row.parent_station_->empty() &&
          stop_ids.contains(row.stop_id_->view())) {
        stop_ids.emplace(row.parent_station_->view());
      }
    }
  }

  {  // Second pass: copy contents.
    auto n_lines = 0U;
    auto reader = utl::make_buf_reader(file_content);
    auto line = reader.read_line();
    auto const header_permutation = utl::read_header<csv_stop>(line);
    out << line.view() << "\n";
    while ((line = reader.read_line())) {
      if (filter_stops) {
        auto const row = utl::read_row<csv_stop>(header_permutation, line);
        if (stop_ids.contains(row.stop_id_->view())) {
          out << line.view() << "\n";
          ++n_lines;
        }
      } else {
        out << line.view() << "\n";
      }
    }
    fmt::println("  stops.txt: lines written: {}", n_lines);
  }
}

void copy_trips(hash_set<std::string> const& trip_ids,
                std::string_view file_content,
                hash_set<std::string>& route_ids,
                hash_set<std::string>& service_ids,
                std::ostream& out) {
  struct csv_trip {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> service_id_;
  };

  auto n_lines = 0U;
  auto reader = utl::make_buf_reader(file_content);
  auto line = reader.read_line();
  auto const header_permutation = utl::read_header<csv_trip>(line);
  out << line.view() << "\n";
  while ((line = reader.read_line())) {
    auto const row = utl::read_row<csv_trip>(header_permutation, line);
    if (trip_ids.contains(row.trip_id_->view())) {
      route_ids.insert(row.route_id_->view());
      service_ids.insert(row.service_id_->view());
      out << line.view() << "\n";
      ++n_lines;
    }
  }

  fmt::println("  trips.txt: lines written: {}", n_lines);
}

void copy_calendar(hash_set<std::string> const& service_ids,
                   std::string_view file_content,
                   std::ostream& out) {
  struct csv_service {
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> service_id_;
  };

  auto n_lines = 0U;
  auto reader = utl::make_buf_reader(file_content);
  auto line = reader.read_line();
  auto const header_permutation = utl::read_header<csv_service>(line);
  out << line.view() << "\n";
  while ((line = reader.read_line())) {
    auto const row = utl::read_row<csv_service>(header_permutation, line);
    if (service_ids.contains(row.service_id_->view())) {
      out << line.view() << "\n";
      ++n_lines;
    }
  }

  fmt::println("  calendar.txt / calendar_dates.txt: lines written: {}",
               n_lines);
}

void copy_routes(hash_set<std::string> const& route_ids,
                 std::string_view file_content,
                 hash_set<std::string>& agency_ids,
                 std::ostream& out) {
  struct csv_service {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id_;
  };

  auto n_lines = 0U;
  auto reader = utl::make_buf_reader(file_content);
  auto line = reader.read_line();
  auto const header_permutation = utl::read_header<csv_service>(line);
  out << line.view() << "\n";
  while ((line = reader.read_line())) {
    auto const row = utl::read_row<csv_service>(header_permutation, line);
    if (route_ids.contains(row.route_id_->view())) {
      agency_ids.insert(row.agency_id_->view());
      out << line.view() << "\n";
      ++n_lines;
    }
  }

  fmt::println("  routes.txt: lines written: {}", n_lines);
}

void copy_agencies(hash_set<std::string> const& agency_ids,
                   std::string_view file_content,
                   std::ostream& out) {
  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id_;
  };

  auto n_lines = 0U;
  auto reader = utl::make_buf_reader(file_content);
  auto line = reader.read_line();
  auto const header_permutation = utl::read_header<csv_stop>(line);
  out << line.view() << "\n";
  while ((line = reader.read_line())) {
    auto const row = utl::read_row<csv_stop>(header_permutation, line);
    if (agency_ids.contains(row.agency_id_->view())) {
      out << line.view() << "\n";
      ++n_lines;
    }
  }

  fmt::println("  agencies.txt: lines written: {}", n_lines);
}

void copy_transfers(hash_set<std::string> const& stop_ids,
                    std::string_view file_content,
                    std::ostream& out) {
  struct csv_stop {
    utl::csv_col<utl::cstr, UTL_NAME("from_stop_id")> from_stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_stop_id")> to_stop_id_;
  };

  auto n_lines = 0U;
  auto reader = utl::make_buf_reader(file_content);
  auto line = reader.read_line();
  auto const header_permutation = utl::read_header<csv_stop>(line);
  out << line.view() << "\n";
  while ((line = reader.read_line())) {
    auto const row = utl::read_row<csv_stop>(header_permutation, line);
    if (stop_ids.contains(row.from_stop_id_->view()) &&
        stop_ids.contains(row.to_stop_id_->view())) {
      out << line.view() << "\n";
      ++n_lines;
    }
  }

  fmt::println("  transfers.txt: lines written: {}", n_lines);
}

int extract(int ac, char** av) {
  auto in = std::vector<fs::path>{"response.json"};
  auto out = fs::path{"gtfs"};
  auto reduce = false;
  auto filter_stops = true;
  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("reduce", po::value(&reduce)->default_value(reduce),
       "Only extract first and last stop of legs for stop times")  //
      ("filter_stops", po::value(&filter_stops)->default_value(filter_stops),
       "Filter stops")  //
      ("in,i", po::value(&in)->multitoken(),
       "PlanResponse JSON input files")  //
      ("out,o", po::value(&out), "output directory");
  auto vm = parse_opt(ac, av, desc);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto important_stops = hash_set<std::string>{};
  auto const add_important_stop = [&](api::Place const& p) {
    if (!reduce || p.vertexType_ != api::VertexTypeEnum::TRANSIT) {
      return;
    }
    auto const tag_end = p.stopId_.value().find('_');
    utl::verify(tag_end != std::string::npos, "no tag found for stop id {}",
                p.stopId_.value());
    auto const [_, added] =
        important_stops.insert(p.stopId_.value().substr(tag_end + 1U));
    if (added) {
      fmt::println("important stop {}", p.stopId_.value().substr(tag_end + 1U));
    }
  };

  auto todos = hash_map<std::string, hash_set<std::string>>{};
  auto source = std::string{};
  auto from_line = std::string{};
  auto to_line = std::string{};
  auto path = std::string{};
  for (auto const& x : in) {
    auto const f =
        cista::mmap{x.generic_string().c_str(), cista::mmap::protection::READ};
    auto const res =
        boost::json::value_to<api::plan_response>(boost::json::parse(f.view()));

    fmt::println("found {} itineraries", res.itineraries_.size());

    for (auto const& i : res.itineraries_) {
      for (auto const& l : i.legs_) {
        add_important_stop(l.from_);
        add_important_stop(l.to_);

        if (!l.source_.has_value() || !l.tripId_.has_value()) {
          continue;
        }

        source.resize(l.source_->size());
        std::reverse_copy(begin(*l.source_), end(*l.source_), begin(source));
        auto const [to, from, p] =
            utl::split<':', utl::cstr, utl::cstr, utl::cstr>(source);

        from_line.resize(from.length());
        std::reverse_copy(begin(from), end(from), begin(from_line));

        to_line.resize(to.length());
        std::reverse_copy(begin(to), end(to), begin(to_line));

        path.resize(p.length());
        std::reverse_copy(begin(p), end(p), begin(path));

        auto const trip_id = split_trip_id(*l.tripId_);

        auto const [_, added] = todos[path].emplace(trip_id.trip_id_);
        if (added) {
          fmt::println("added {}:{}:{}, trip_id={}", path, from_line, to_line,
                       trip_id.trip_id_);
        }
      }
    }
  }

  auto stop_ids = hash_set<std::string>{};
  auto route_ids = hash_set<std::string>{};
  auto service_ids = hash_set<std::string>{};
  auto agency_ids = hash_set<std::string>{};
  for (auto const& [stop_times_str, trip_ids] : todos) {
    auto const stop_times_path = fs::path{stop_times_str};

    stop_ids.clear();
    route_ids.clear();
    service_ids.clear();
    agency_ids.clear();

    utl::verify(stop_times_path.filename() == "stop_times.txt",
                "expected filename stop_times.txt, got \"{}\"",
                stop_times_path);

    auto const dataset_dir = stop_times_path.parent_path();
    utl::verify(stop_times_path.has_parent_path() && fs::exists(dataset_dir),
                "expected path \"{}\" to have existent parent path",
                stop_times_path);

    auto const dir = n::loader::make_dir(dataset_dir);
    utl::verify(dir->exists("stop_times.txt"),
                "no stop_times.txt file found in {}", dataset_dir);

    auto ec = std::error_code{};
    fs::create_directories(out / dataset_dir.filename(), ec);

    {
      fmt::println("writing {}/stop_times.txt, searching for trips={}",
                   out / dataset_dir.filename(), trip_ids);
      auto of = std::ofstream{out / dataset_dir.filename() / "stop_times.txt"};
      fmt::println("important stops: {}", important_stops);
      copy_stop_times(trip_ids, important_stops,
                      dir->get_file("stop_times.txt").data(), stop_ids, of);
    }

    {
      fmt::println("writing {}/stops.txt, searching for stops={}",
                   out / dataset_dir.filename(), stop_ids);
      auto of = std::ofstream{out / dataset_dir.filename() / "stops.txt"};
      copy_stops(stop_ids, dir->get_file("stops.txt").data(), of, filter_stops);
    }

    {
      fmt::println("writing {}/trips.txt", out / dataset_dir.filename());
      auto of = std::ofstream{out / dataset_dir.filename() / "trips.txt"};
      copy_trips(trip_ids, dir->get_file("trips.txt").data(), route_ids,
                 service_ids, of);
    }

    {
      fmt::println("writing {}/routes.txt, searching for routes={}",
                   out / dataset_dir.filename(), route_ids);
      auto of = std::ofstream{out / dataset_dir.filename() / "routes.txt"};
      copy_routes(route_ids, dir->get_file("routes.txt").data(), agency_ids,
                  of);
    }

    if (dir->exists("calendar.txt")) {
      fmt::println("writing {}/calendar.txt, searching for service_ids={}",
                   out / dataset_dir.filename(), service_ids);
      auto of = std::ofstream{out / dataset_dir.filename() / "calendar.txt"};
      copy_calendar(service_ids, dir->get_file("calendar.txt").data(), of);
    }

    if (dir->exists("calendar_dates.txt")) {
      fmt::println(
          "writing {}/calendar_dates.txt, searching for service_ids={}",
          out / dataset_dir.filename(), service_ids);
      auto of =
          std::ofstream{out / dataset_dir.filename() / "calendar_dates.txt"};
      copy_calendar(service_ids, dir->get_file("calendar_dates.txt").data(),
                    of);
    }

    if (dir->exists("agency.txt")) {
      fmt::println("writing {}/agency.txt, searching for agencies={}",
                   out / dataset_dir.filename(), agency_ids);
      auto of = std::ofstream{out / dataset_dir.filename() / "agency.txt"};
      copy_agencies(agency_ids, dir->get_file("agency.txt").data(), of);
    }

    if (dir->exists("transfers.txt")) {
      fmt::println("writing {}/transfers.txt", out / dataset_dir.filename());
      auto of = std::ofstream{out / dataset_dir.filename() / "transfers.txt"};
      copy_transfers(stop_ids, dir->get_file("transfers.txt").data(), of);
    }

    if (dir->exists("feed_info.txt")) {
      std::ofstream{out / dataset_dir.filename() / "feed_info.txt"}
          << dir->get_file("feed_info.txt").data();
    }
  }

  return 0;
}

}  // namespace motis
