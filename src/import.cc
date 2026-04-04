#include "motis/import.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <vector>

#include "nigiri/clasz.h"

#include "motis/clog_redirect.h"
#include "motis/data.h"
#include "motis/hashes.h"
#include "motis/import/adr_extend_import.h"
#include "motis/import/adr_import.h"
#include "motis/import/dataset_hashes.h"
#include "motis/import/matches_import.h"
#include "motis/import/osr_footpath_import.h"
#include "motis/import/osr_import.h"
#include "motis/import/route_shapes_import.h"
#include "motis/import/task.h"
#include "motis/import/tbd_import.h"
#include "motis/import/tiles_import.h"
#include "motis/import/tt_import.h"
#include "motis/import/way_matches_import.h"

#include "utl/verify.h"

namespace fs = std::filesystem;

namespace motis {

void import(config const& c, fs::path const& data_path) {
  c.verify_input_files_exist();

  auto ec = std::error_code{};
  fs::create_directories(data_path / "logs", ec);
  fs::create_directories(data_path / "meta", ec);
  {
    auto cfg = std::ofstream{(data_path / "config.yml").generic_string()};
    cfg.exceptions(std::ios_base::badbit | std::ios_base::eofbit);
    cfg << c << "\n";
    cfg.close();
  }

  clog_redirect::set_enabled(true);

  auto const hashes = dataset_hashes{c};

  auto tiles = tiles_import{data_path, c, hashes};
  auto osr = osr_import{data_path, c, hashes};
  auto adr = adr_import{data_path, c, hashes};
  auto tt = tt_import{data_path, c, hashes};
  auto tbd = tbd_import{data_path, c, hashes};
  auto adr_extend = adr_extend_import{data_path, c, hashes};
  auto osr_footpath = osr_footpath_import{data_path, c, hashes};
  auto matches = matches_import{data_path, c, hashes};
  auto way_matches = way_matches_import{data_path, c, hashes};
  auto route_shapes = route_shapes_import{data_path, c, hashes};

  tbd.add_dependency({tt});
  adr_extend.add_dependency({adr, tt});
  osr_footpath.add_dependency({osr, tt});
  if (c.osr_footpath_) {
    matches.add_dependency({osr_footpath});
  } else {
    matches.add_dependency({osr, tt});
  }
  way_matches.add_dependency({matches});
  route_shapes.add_dependency({osr, tt});

  auto tasks = std::vector<task*>{&tiles,        &osr,     &adr,
                                  &tt,           &tbd,     &adr_extend,
                                  &osr_footpath, &matches, &way_matches,
                                  &route_shapes};
  std::erase_if(tasks, [](task* const t) { return !t->is_enabled(); });
  std::erase_if(tasks, [](task* const t) { return t->can_load(); });

  while (!tasks.empty()) {
    auto const task_it = std::ranges::find_if(
        tasks, [](task const* t) { return t->is_ready_to_run(); });
    utl::verify(task_it != end(tasks), "no task to run");
    (*task_it)->exec();
    tasks.erase(task_it);
  }
}

}  // namespace motis
