#include "motis/bootstrap/import_files.h"

#include <algorithm>
#include <iostream>
#include <regex>
#include <vector>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "utl/pipes.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;
namespace mi = motis::import;
namespace ml = motis::logging;
namespace mm = motis::module;

namespace motis::bootstrap {

mm::msg_ptr make_file_event(std::vector<std::string> const& import_paths) {
  std::regex re{R"(^(\w+)(?:\[(.*?)\])?:(.*)$)"};

  mm::message_creator mc;
  std::vector<flatbuffers::Offset<mi::ImportPath>> fbs_paths;
  for (auto const& import_path : import_paths) {
    std::smatch m;
    utl::verify(std::regex_match(import_path, m, re) && m.size() == 4,
                "import_path does not match tag[options]:path : {}",
                import_path);

    utl::verify(fs::exists(m.str(3)), "file does not exist: {}", m.str(3));

    fbs_paths.push_back(mi::CreateImportPath(mc, mc.CreateString(m.str(1)),
                                             mc.CreateString(m.str(2)),
                                             mc.CreateString(m.str(3))));
  }
  mc.create_and_finish(
      MsgContent_FileEvent,
      mi::CreateFileEvent(mc, mc.CreateVector(fbs_paths)).Union(), "/import",
      DestinationType_Topic);

  return make_msg(mc);
}

std::vector<std::string> pick_files(mm::msg_ptr const& msg,
                                    std::string const& tag) {
  using mi::FileEvent;
  return utl::all(*motis_content(FileEvent, msg)->paths())  //
         | utl::remove_if([&](auto&& p) { return p->tag()->str() == tag; })  //
         | utl::transform([](auto&& p) { return p->path()->str(); })  //
         | utl::remove_if([](auto&& p) { return !fs::is_regular_file(p); })  //
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

void import_coastline(mm::msg_ptr const& msg) {
  auto const paths = pick_files(msg, "coastline");
  if (paths.size() > 1) {
    LOG(ml::warn) << "import_coastline: more than one file matches!";
  }
  for (auto const& path : paths) {
    mm::message_creator mc;
    mc.create_and_finish(
        MsgContent_CoastlineEvent,
        mi::CreateCoastlineEvent(mc, mc.CreateString(path), file_hash(path),
                                 fs::file_size(path))
            .Union(),
        "/import", DestinationType_Topic);
    motis_publish(make_msg(mc));
  }
}

void import_osm(mm::msg_ptr const& msg) {
  auto const paths = pick_files(msg, "osm");
  if (paths.size() > 1) {
    LOG(ml::warn) << "import_osm: more than one file matches!";
  }
  for (auto const& path : paths) {
    mm::message_creator mc;
    mc.create_and_finish(
        MsgContent_OSMEvent,
        mi::CreateOSMEvent(mc, mc.CreateString(path), file_hash(path),
                           fs::file_size(path))
            .Union(),
        "/import", DestinationType_Topic);
    motis_publish(make_msg(mc));
  }
}

void import_dem(mm::msg_ptr const& msg) {
  using motis::import::FileEvent;
  for (auto const* ip : *motis_content(FileEvent, msg)->paths()) {
    auto const path = ip->path()->str();
    if (ip->tag()->str() != "dem" || !fs::is_directory(path)) {
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

    mm::message_creator mc;
    mc.create_and_finish(
        MsgContent_DEMEvent,
        mi::CreateDEMEvent(mc, mc.CreateString(path), hash).Union(), "/import",
        DestinationType_Topic);
    ctx::await_all(motis_publish(make_msg(mc)));
  }
}

void register_import_files(mm::registry& r) {
  r.subscribe("/import", [](mm::msg_ptr const& msg) {
    if (msg->get()->content_type() == MsgContent_FileEvent) {
      import_osm(msg);
      import_coastline(msg);
      import_dem(msg);
    }
    return nullptr;
  });
}

}  // namespace motis::bootstrap
