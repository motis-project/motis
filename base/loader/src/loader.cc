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

#include "utl/parser/file.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"
#include "utl/visit.h"

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
  auto const mem =
      utl::to_vec(opt.dataset_, [&](auto const& path) -> dataset_mem_t {
        auto const binary_schedule_file = fs::path(path) / SCHEDULE_FILE;
        if (fs::is_regular_file(binary_schedule_file)) {
          return cista::mmap{binary_schedule_file.generic_string().c_str(),
                             cista::mmap::protection::READ};
        }

        for (auto const& parser : parsers()) {
          if (parser->applicable(path)) {
            flatbuffers64::FlatBufferBuilder builder;
            parser->parse(path, builder);
            if (opt.write_serialized_) {
              utl::file(binary_schedule_file.string().c_str(), "w+")
                  .write(builder.GetBufferPointer(), builder.GetSize());
            }
            return typed_flatbuffer<Schedule>{std::move(builder)};
          }
        }

        for (auto const& parser : parsers()) {
          std::clog << "missing files:\n";
          for (auto const& file : parser->missing_files(path)) {
            std::clog << "  " << file << "\n";
          }
        }
        throw utl::fail("no parser for dataset {}", path);
      });

  auto const datasets = utl::to_vec(mem, [&](dataset_mem_t const& v) {
    return std::visit(
        overloaded{[](cista::mmap const& m) -> Schedule const* {
                     return GetSchedule(m.data());
                   },
                   [](typed_flatbuffer<Schedule> const& m) -> Schedule const* {
                     return m.get();
                   }},
        v);
  });

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
