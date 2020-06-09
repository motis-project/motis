#include "motis/bootstrap/import_coastline.h"

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "motis/module/context/motis_publish.h"

namespace fs = boost::filesystem;
using motis::module::message_creator;
using motis::module::msg_ptr;

namespace motis::bootstrap {

msg_ptr import_coastline(msg_ptr const& msg) {
  if (msg->get()->content_type() != MsgContent_FileEvent) {
    return nullptr;
  }

  using motis::import::FileEvent;
  auto const path = motis_content(FileEvent, msg)->path()->str();
  auto const name = fs::path{path}.filename().generic_string();
  if (!fs::is_regular_file(path) || !boost::ends_with(name, ".zip")) {
    return nullptr;
  }

  auto hash = cista::hash_t{};
  auto size = std::size_t{};
  {
    cista::mmap m{path.c_str(), cista::mmap::protection::READ};
    hash = cista::hash(std::string_view{
        reinterpret_cast<char const*>(m.begin()),
        std::min(static_cast<size_t>(50 * 1024 * 1024), m.size())});
    size = m.size();
  }

  message_creator fbb;
  fbb.create_and_finish(MsgContent_CoastlineEvent,
                        motis::import::CreateCoastlineEvent(
                            fbb, fbb.CreateString(path), hash, size)
                            .Union(),
                        "/import", DestinationType_Topic);
  motis_publish(make_msg(fbb));
  return nullptr;
}

}  // namespace motis::bootstrap
