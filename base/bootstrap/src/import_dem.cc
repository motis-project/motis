#include "motis/bootstrap/import_dem.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "boost/filesystem.hpp"

#include "cista/hash.h"

#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;
using motis::module::message_creator;
using motis::module::msg_ptr;

namespace motis::bootstrap {

struct dem_file {
  std::string name_;
  uint64_t size_;
};

msg_ptr import_dem(msg_ptr const& msg) {
  if (msg->get()->content_type() != MsgContent_FileEvent) {
    return nullptr;
  }

  using motis::import::FileEvent;
  auto const path = motis_content(FileEvent, msg)->path()->str();
  if (!fs::is_directory(path)) {
    return nullptr;
  }

  auto dem_files = std::vector<dem_file>{};
  auto const hdr_ext = fs::path{".hdr"};
  auto const bil_ext = fs::path{".bil"};
  for (auto const& entry : fs::directory_iterator{path}) {
    auto const& p = entry.path();
    if (fs::is_regular_file(p) &&
        (p.extension() == hdr_ext || p.extension() == bil_ext)) {
      dem_files.emplace_back(dem_file{p.filename().string(), fs::file_size(p)});
    }
  }
  if (dem_files.empty()) {
    return nullptr;
  }
  std::sort(begin(dem_files), end(dem_files),
            [](auto const& a, auto const& b) { return a.name_ < b.name_; });

  auto hash = cista::BASE_HASH;
  for (auto const& file : dem_files) {
    hash = cista::hash(file.name_, hash);
    hash = cista::hash_combine(hash, file.size_);
  }

  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_DEMEvent,
      motis::import::CreateDEMEvent(fbb, fbb.CreateString(path), hash).Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
  return nullptr;
}

}  // namespace motis::bootstrap
