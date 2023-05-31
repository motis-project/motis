#include "motis/loader/loader.h"

#include <memory>
#include <ostream>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <filesystem>

#include "flatbuffers/flatbuffers.h"

#include "cista/mmap.h"

#include "utl/enumerate.h"
#include "utl/overloaded.h"
#include "utl/parser/file.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/common/typed_flatbuffer.h"
#include "motis/core/schedule/serialization.h"

#include "motis/loader/build_graph.h"
#include "motis/loader/gtfs/gtfs_parser.h"
#include "motis/loader/hrd/hrd_parser.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace fs = std::filesystem;
namespace ml = motis::logging;

namespace motis::loader {

std::vector<std::unique_ptr<format_parser>> parsers() {
  std::vector<std::unique_ptr<format_parser>> p;
  p.emplace_back(std::make_unique<gtfs::gtfs_parser>());
  p.emplace_back(std::make_unique<hrd::hrd_parser>());
  return p;
}

using dataset_mem_t = std::variant<cista::mmap, typed_flatbuffer<Schedule>>;

schedule_ptr load_schedule_impl(loader_options const& loader_opt,
                                cista::memory_holder& schedule_buf,
                                std::string const& data_dir) {
  ml::scoped_timer const time("loading schedule");

  // ensure there is an active progress tracker (e.g. for test cases)
  utl::get_active_progress_tracker_or_activate("schedule");

  auto const graph_path = loader_opt.graph_path(data_dir);
  auto enable_read_graph = loader_opt.read_graph_;
  auto enable_write_graph = loader_opt.write_graph_;
  if (loader_opt.cache_graph_) {
    enable_read_graph = fs::is_regular_file(graph_path);
    enable_write_graph = true;
  }
  if (enable_read_graph) {
    utl::verify(fs::is_regular_file(graph_path), "graph not found: {}",
                graph_path);
    LOG(ml::info) << "reading graph: " << graph_path;
    try {
      return read_graph(graph_path, schedule_buf, loader_opt.read_graph_mmap_);
    } catch (std::runtime_error const& err) {
      if (loader_opt.cache_graph_) {
        LOG(ml::info) << "could not load existing graph, updating cache ("
                      << err.what() << ")";
      } else {
        throw err;
      }
    }
  }

  utl::verify(!loader_opt.dataset_.empty(),
              "load_schedule: loader_opt.dataset_.empty()");
  utl::verify(
      loader_opt.dataset_.size() == 1 ||
          loader_opt.dataset_.size() == loader_opt.dataset_prefix_.size(),
      "load_schedule: dataset/prefix size mismatch");

  std::vector<dataset_mem_t> mem;
  mem.reserve(loader_opt.dataset_.size());
  for (auto const& [i, path] : utl::enumerate(loader_opt.dataset_)) {
    auto const binary_schedule_file = loader_opt.fbs_schedule_path(data_dir, i);
    if (fs::is_regular_file(binary_schedule_file)) {
      mem.emplace_back(cista::mmap{binary_schedule_file.c_str(),
                                   cista::mmap::protection::READ});
      continue;
    }

    auto const all_parsers = parsers();
    auto const it = std::find_if(
        begin(all_parsers), end(all_parsers),
        [&p = path](auto const& parser) { return parser->applicable(p); });

    if (it == end(all_parsers)) {
      for (auto const& parser : parsers()) {
        std::clog << "missing files:\n";
        for (auto const& file : parser->missing_files(path)) {
          std::clog << "  " << file << "\n";
        }
      }
      throw utl::fail("no parser for dataset {}", path);
    }

    auto progress_tracker = utl::activate_progress_tracker(
        loader_opt.dataset_prefix_.empty() ||
                loader_opt.dataset_prefix_[i].empty()
            ? fmt::format("parse {}", i)
            : fmt::format("parse {}", loader_opt.dataset_prefix_[i]));

    flatbuffers64::FlatBufferBuilder builder;
    try {
      (**it).parse({loader_opt.link_stop_distance_}, path, builder);
      progress_tracker->status("FINISHED").show_progress(false);
    } catch (std::exception const& e) {
      progress_tracker->status(fmt::format("ERROR: {}", e.what()))
          .show_progress(false);
      throw;
    } catch (...) {
      progress_tracker->status("ERROR: UNKNOWN EXCEPTION").show_progress(false);
      throw;
    }

    if (loader_opt.write_serialized_) {
      auto const schedule_dir = fs::path{binary_schedule_file}.parent_path();
      if (!schedule_dir.empty()) {
        fs::create_directories(schedule_dir);
      }
      utl::file(binary_schedule_file.c_str(), "w+")
          .write(builder.GetBufferPointer(), builder.GetSize());
    }

    mem.emplace_back(typed_flatbuffer<Schedule>{std::move(builder)});
  }

  auto const datasets = utl::to_vec(mem, [&](dataset_mem_t const& v) {
    return std::visit(
        utl::overloaded{[](cista::mmap const& m) -> Schedule const* {
                          return GetSchedule(m.data());
                        },
                        [](typed_flatbuffer<Schedule> const& m)
                            -> Schedule const* { return m.get(); }},
        v);
  });

  utl::activate_progress_tracker("schedule");
  auto sched = build_graph(datasets, loader_opt);
  if (enable_write_graph) {
    LOG(ml::info) << "writing graph: " << graph_path;
    auto const graph_dir = fs::path{graph_path}.parent_path();
    if (!graph_dir.empty()) {
      fs::create_directories(graph_dir);
    }
    write_graph(graph_path, *sched);
  }
  return sched;
}

#ifdef _WIN32
bool load_schedule_checked(loader_options const& opt,
                           cista::memory_holder& schedule_buf,
                           std::string const& data_dir, schedule_ptr& ptr) {
  __try {
    [&]() { ptr = load_schedule_impl(opt, schedule_buf, data_dir); }();
    return true;
  } __except (GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR
                  ? EXCEPTION_EXECUTE_HANDLER
                  : EXCEPTION_CONTINUE_SEARCH) {
    return false;
  }
}
#endif

schedule_ptr load_schedule(loader_options const& loader_opt,
                           cista::memory_holder& schedule_buf,
                           std::string const& data_dir) {
#ifdef _WIN32
  auto ptr = schedule_ptr{};
  utl::verify(load_schedule_checked(loader_opt, schedule_buf, data_dir, ptr),
              "load_schedule: file access error: EXCEPTION_IN_PAGE_ERROR");
  return ptr;
#else
  return load_schedule_impl(loader_opt, schedule_buf, data_dir);
#endif
}

schedule_ptr load_schedule(loader_options const& loader_opt) {
  utl::verify(!loader_opt.read_graph_,
              "load_schedule: read_graph requires buffer");
  cista::memory_holder buf{};
  return load_schedule(loader_opt, buf, "");
}

}  // namespace motis::loader
