#include "motis/gbfs/system_information.h"

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "motis/json/json.h"

using namespace motis::json;

namespace motis::gbfs {

system_information read_system_information(std::string_view s) {
  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    throw utl::fail("GBFS system_information: Bad JSON: {} at offset {}",
                    rapidjson::GetParseError_En(doc.GetParseError()),
                    doc.GetErrorOffset());
  }

  auto const& data = get_obj(doc, "data");
  auto const read_key = [&](char const* key) -> std::string {
    if (auto const it = data.FindMember(key);
        it != data.MemberEnd() && it->value.IsString()) {
      return {it->value.GetString(), it->value.GetStringLength()};
    }
    return {};
  };

  return system_information{.name_ = read_key("name"),
                            .name_short_ = read_key("short_name"),
                            .operator_ = read_key("operator"),
                            .url_ = read_key("url"),
                            .purchase_url_ = read_key("purchase_url"),
                            .mail_ = read_key("email")};
}

}  // namespace motis::gbfs
