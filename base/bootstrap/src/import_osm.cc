#include "motis/bootstrap/import_osm.h"

#include "boost/filesystem.hpp"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;
using motis::module::message_creator;
using motis::module::msg_ptr;

namespace motis::bootstrap {

motis::module::msg_ptr import_osm(motis::module::msg_ptr const& msg) {
  if (msg->get()->content_type() != MsgContent_FileEvent) {
    return nullptr;
  }

  using motis::import::FileEvent;
  auto const path = motis_content(FileEvent, msg)->path()->str();
  auto const name = fs::path{path}.filename().generic_string();
  if (!fs::is_regular_file(path) || name.length() < 8 ||
      name.substr(name.size() - std::min(size_t{8U}, name.size())) !=
          ".osm.pbf") {
    return nullptr;
  }
  cista::mmap m(path.c_str(), cista::mmap::protection::READ);
  auto const hash = cista::hash(std::string_view{
      reinterpret_cast<char const*>(m.begin()),
      std::min(static_cast<size_t>(50 * 1024 * 1024), m.size())});

  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_OSMEvent,
      motis::import::CreateOSMEvent(fbb, fbb.CreateString(path), hash, m.size())
          .Union(),
      "/import", DestinationType_Topic);
  ctx::await_all(motis_publish(make_msg(fbb)));
  return nullptr;
}

}  // namespace motis::bootstrap
