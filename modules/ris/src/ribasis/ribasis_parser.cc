#include "motis/ris/ribasis/ribasis_parser.h"

#include <cstdint>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"

#include "utl/verify.h"

#include "motis/json/json.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/event_type.h"

#include "motis/ris/ribasis/common.h"
#include "motis/ris/ribasis/ribasis_fahrt_parser.h"
#include "motis/ris/ribasis/ribasis_formation_parser.h"

using namespace flatbuffers;
using namespace motis::logging;
using namespace motis::json;

namespace motis::ris::ribasis {

bool to_ris_message(std::string_view s,
                    const std::function<void(ris_message&&)>& cb,
                    std::string const& tag) {
  utl::verify(tag.empty(), "RI Basis does not support multi-schedule");

  rapidjson::Document doc;
  if (doc.Parse(s.data(), s.size()).HasParseError()) {
    doc.GetParseError();
    LOG(error) << "RI Basis: Bad JSON: "
               << rapidjson::GetParseError_En(doc.GetParseError())
               << " at offset " << doc.GetErrorOffset();
    return false;
  }

  try {
    utl::verify(doc.IsObject(), "no root object");
    auto const& meta = get_obj(doc, "meta");
    auto const& data = get_obj(doc, "data");
    auto const created_at = get_timestamp(meta, "created", "%FT%H:%M:%S%Ez");
    auto ctx = ris_msg_context{created_at};

    if (has_key(data, "fahrtid")) {
      fahrt::parse_ribasis_fahrt(ctx, data);
    } else if (has_key(data, "fahrt")) {
      formation::parse_ribasis_formation(ctx, data);
    } else {
      LOG(error) << "invalid/unsupported RI Basis message";
      return false;
    }

    utl::verify(ctx.earliest_ != std::numeric_limits<unixtime>::max(),
                "earliest not set");
    utl::verify(ctx.latest_ != std::numeric_limits<unixtime>::min(),
                "latest not set");
    cb(ris_message{ctx.earliest_, ctx.latest_, ctx.timestamp_,
                   std::move(ctx.b_)});
  } catch (std::runtime_error const& e) {
    LOG(error) << "unable to parse RI Basis message: " << e.what();
    return false;
  }
  return true;
}

std::vector<ris_message> parse(std::string_view s, std::string const& tag) {
  utl::verify(tag.empty(), "RI Basis does not support multi-schedule");
  std::vector<ris_message> msgs;
  to_ris_message(
      s, [&](ris_message&& m) { msgs.emplace_back(std::move(m)); }, tag);
  return msgs;
}

}  // namespace motis::ris::ribasis
