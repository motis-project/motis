#include "motis/loader/loader.h"

#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <variant>
#include <vector>

#include "cista/serialization.h"
#include "cista/targets/file.h"

#include "boost/filesystem.hpp"

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

namespace fs = boost::filesystem;
namespace ml = motis::logging;

namespace motis::loader {

std::vector<std::unique_ptr<format_parser>> parsers() {
  std::vector<std::unique_ptr<format_parser>> p;
  p.emplace_back(std::make_unique<gtfs::gtfs_parser>());
  p.emplace_back(std::make_unique<hrd::hrd_parser>());
  return p;
}

using dataset_mem_t = std::variant<cista::mmap, typed_flatbuffer<Schedule>>;

schedule_ptr load_schedule(loader_options const& opt,
                           cista::memory_holder& schedule_buf) {
  ml::scoped_timer time("loading schedule");

  // ensure there is an active progress tracker (e.g. for test cases)
  utl::get_active_progress_tracker_or_activate("schedule");

  auto const graph_path = opt.graph_path();
  if (opt.read_graph_) {
    utl::verify(fs::is_regular_file(graph_path), "graph not found");
    LOG(ml::info) << "reading graph " << graph_path;
    return read_graph(graph_path, schedule_buf, opt.read_graph_mmap_);
  }

  utl::verify(!opt.dataset_.empty(), "load_schedule: opt.dataset_.empty()");
  utl::verify(opt.dataset_.size() == 1 ||
                  opt.dataset_.size() == opt.dataset_prefix_.size(),
              "load_schedule: dataset/prefix size mismatch");

  std::vector<dataset_mem_t> mem;
  mem.reserve(opt.dataset_.size());
  for (auto const& [i, path] : utl::enumerate(opt.dataset_)) {
    auto const binary_schedule_file = fs::path(path) / SCHEDULE_FILE;
    if (fs::is_regular_file(binary_schedule_file)) {
      mem.emplace_back(
          cista::mmap{binary_schedule_file.generic_string().c_str(),
                      cista::mmap::protection::READ});
      continue;
    }

    auto const all_parsers = parsers();
    auto const it = std::find_if(
        begin(all_parsers), end(all_parsers),
        [& p = path](auto const& parser) { return parser->applicable(p); });

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
        opt.dataset_prefix_.empty() || opt.dataset_prefix_[i].empty()
            ? std::string{"parse {}", i}
            : fmt::format("parse {}", opt.dataset_prefix_[i]));

    flatbuffers64::FlatBufferBuilder builder;
    try {
      (**it).parse(path, builder);
      progress_tracker->status("FINISHED").show_progress(false);
    } catch (std::exception const& e) {
      progress_tracker->status(fmt::format("ERROR: {}", e.what()))
          .show_progress(false);
      throw;
    } catch (...) {
      progress_tracker->status("ERROR: UNKNOWN EXCEPTION").show_progress(false);
      throw;
    }

    if (opt.write_serialized_) {
      utl::file(binary_schedule_file.string().c_str(), "w+")
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
  auto sched = build_graph(datasets, opt);
  if (opt.write_graph_) {
    write_graph(graph_path, *sched);
  }
  return sched;
}

schedule_ptr load_schedule(loader_options const& opt) {
  utl::verify(!opt.read_graph_, "load_schedule: read_graph requires buffer");
  cista::memory_holder buf{};
  return load_schedule(opt, buf);
}

}  // namespace motis::loader
