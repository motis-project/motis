#include "motis/loader/loader.h"

#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <vector>

#include "cista/serialization.h"
#include "cista/targets/file.h"

#include "boost/filesystem.hpp"

#include "flatbuffers/flatbuffers.h"

#include "cista/mmap.h"

#include "utl/parser/file.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/serialization.h"
#include "motis/loader/build_graph.h"
#include "motis/loader/gtfs/gtfs_parser.h"
#include "motis/loader/hrd/hrd_parser.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace fs = boost::filesystem;
using namespace flatbuffers64;
using namespace utl;
using namespace motis::logging;

namespace motis::loader {

std::vector<std::unique_ptr<format_parser>> parsers() {
  std::vector<std::unique_ptr<format_parser>> p;
  p.emplace_back(std::make_unique<gtfs::gtfs_parser>());
  p.emplace_back(std::make_unique<hrd::hrd_parser>());
  return p;
}

schedule_ptr load_schedule(loader_options const& opt,
                           cista::memory_holder& schedule_buf) {
  scoped_timer time("loading schedule");

  auto const binary_schedule_file = fs::path(opt.dataset_) / SCHEDULE_FILE;
  auto const serialized_file_path = opt.graph_path();

  if (opt.read_graph_) {
    utl::verify(fs::is_regular_file(serialized_file_path), "graph not found");
    LOG(info) << "reading graph " << serialized_file_path;
    return read_graph(serialized_file_path, schedule_buf, opt.read_graph_mmap_);
  }

  // ensure there is an active progress tracker (e.g. for test cases)
  utl::get_active_progress_tracker_or_activate("schedule");

  if (fs::is_regular_file(binary_schedule_file)) {
    auto buf = file(binary_schedule_file.string().c_str(), "r").content();
    auto sched = build_graph(GetSchedule(buf.buf_), opt);
    if (opt.write_graph_) {
      write_graph(serialized_file_path, *sched);
    }
    return sched;
  } else {
    for (auto const& parser : parsers()) {
      if (parser->applicable(opt.dataset_)) {
        FlatBufferBuilder builder;
        parser->parse(opt.dataset_, builder);
        if (opt.write_serialized_) {
          utl::get_active_progress_tracker()->status("Write Schedule File");
          utl::file(binary_schedule_file.string().c_str(), "w+")
              .write(builder.GetBufferPointer(), builder.GetSize());
        }
        auto sched =
            build_graph(GetSchedule(builder.GetBufferPointer()), opt, 80);
        if (opt.write_graph_) {
          write_graph(serialized_file_path, *sched);
        }
        return sched;
      }
    }

    for (auto const& parser : parsers()) {
      std::clog << "missing files:\n";
      for (auto const& file : parser->missing_files(opt.dataset_)) {
        std::clog << "  " << file << "\n";
      }
    }
    throw std::runtime_error("no parser was applicable");
  }
}

schedule_ptr load_schedule(loader_options const& opt) {
  utl::verify(!opt.read_graph_, "load_schedule: read_graph requires buffer");
  cista::memory_holder buf{};
  return load_schedule(opt, buf);
}

}  // namespace motis::loader
