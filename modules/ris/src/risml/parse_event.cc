#include "motis/ris/risml/parse_event.h"

#include "motis/ris/risml/common.h"
#include "motis/ris/risml/parse_station.h"
#include "motis/ris/risml/parse_time.h"
#include "motis/ris/risml/parse_type.h"

using namespace flatbuffers;
using namespace pugi;

namespace motis::ris::risml {

Offset<AdditionalEvent> parse_additional_event(FlatBufferBuilder& fbb,
                                               Offset<Event> const& event,
                                               xml_node const& e_node,
                                               xml_node const& t_node) {
  auto track_attr = child_attr(e_node, "Gleis", "Soll");
  auto track =
      fbb.CreateString(track_attr != nullptr ? track_attr.value() : "");
  auto category = fbb.CreateString(t_node.attribute("Gattung").value());
  return CreateAdditionalEvent(fbb, event, category, track);
}

boost::optional<Offset<Event>> parse_standalone_event(context& ctx,
                                                      xml_node const& e_node) {
  auto event_type = parse_type(e_node.attribute("Typ").value());
  if (event_type == boost::none) {
    return boost::none;
  }

  auto station = parse_station(ctx.b_, e_node);
  auto service_num = child_attr(e_node, "Zug", "Nr").as_uint();
  auto line_id = child_attr(e_node, "Zug", "Linie").value();
  auto line_id_offset = ctx.b_.CreateString(line_id);

  auto schedule_time =
      parse_schedule_time(ctx, child_attr(e_node, "Zeit", "Soll").value());

  return CreateEvent(ctx.b_, station, service_num, line_id_offset, *event_type,
                     schedule_time);
}

}  // namespace motis::ris::risml
