#include "motis/bootstrap/import_files.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "utl/pipes.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;
using motis::module::message_creator;
using motis::module::msg_ptr;

namespace motis::bootstrap {

template <typename Pred>
std::vector<std::string> pick_files(msg_ptr const& msg, Pred&& pred) {
  using motis::import::FileEvent;
  return utl::all(*motis_content(FileEvent, msg)->paths())  //
         | utl::transform([](auto&& p) { return p->str(); })  //
         | utl::remove_if([&](auto&& p) {
             return !fs::is_regular_file(p) || !pred(p);
           })  //
         | utl::vec();
}

cista::hash_t file_hash(std::string const& path) {
  auto hash = cista::BASE_HASH;
  cista::mmap m{path.c_str(), cista::mmap::protection::READ};
  hash = cista::hash(
      std::string_view{
          reinterpret_cast<char const*>(m.begin()),
          std::min(static_cast<size_t>(50 * 1024 * 1024), m.size())},
      hash);
  hash = cista::hash_combine(hash, m.size());
  return hash;
}

void import_coastline(msg_ptr const& msg) {
  auto const paths = pick_files(
      msg, [](auto const& path) { return boost::ends_with(path, ".zip"); });
  if (paths.size() > 1) {
    LOG(logging::warn) << "import_coastline: more than one file matches!";
  }
  for (auto const& path : paths) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_CoastlineEvent,
        motis::import::CreateCoastlineEvent(
            fbb, fbb.CreateString(path), file_hash(path), fs::file_size(path))
            .Union(),
        "/import", DestinationType_Topic);
    motis_publish(make_msg(fbb));
  }
}

void import_osm(msg_ptr const& msg) {
  auto const paths = pick_files(
      msg, [](auto const& path) { return boost::ends_with(path, ".osm.pbf"); });
  if (paths.size() > 1) {
    LOG(logging::warn) << "import_osm: more than one file matches!";
  }
  for (auto const& path : paths) {
    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_OSMEvent,
        motis::import::CreateOSMEvent(fbb, fbb.CreateString(path),
                                      file_hash(path), fs::file_size(path))
            .Union(),
        "/import", DestinationType_Topic);
    motis_publish(make_msg(fbb));
  }
}

void import_dem(msg_ptr const& msg) {
  using motis::import::FileEvent;
  for (auto const* p_str : *motis_content(FileEvent, msg)->paths()) {
    auto const path = p_str->str();
    if (!fs::is_directory(path)) {
      continue;
    }
    auto dem_files = std::vector<std::string>{};
    auto const hdr_ext = fs::path{".hdr"};
    auto const bil_ext = fs::path{".bil"};
    for (auto const& entry : fs::directory_iterator{path}) {
      auto const& p = entry.path();
      if (fs::is_regular_file(p) &&
          (p.extension() == hdr_ext || p.extension() == bil_ext)) {
        dem_files.emplace_back(p.filename().string());
      }
    }
    if (dem_files.empty()) {
      continue;
    }
    std::sort(begin(dem_files), end(dem_files));

    auto hash = cista::BASE_HASH;
    for (auto const& p : dem_files) {
      hash = cista::hash(p, hash);
      hash = cista::hash_combine(hash, fs::file_size(p));
    }

    message_creator fbb;
    fbb.create_and_finish(
        MsgContent_DEMEvent,
        motis::import::CreateDEMEvent(fbb, fbb.CreateString(path), hash)
            .Union(),
        "/import", DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(fbb)));
  }
}

motis::module::msg_ptr import_files(motis::module::msg_ptr const& msg) {
  if (msg->get()->content_type() == MsgContent_FileEvent) {
    import_osm(msg);
    import_coastline(msg);
    import_dem(msg);
  }

  return nullptr;
}

}  // namespace motis::bootstrap
