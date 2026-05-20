#include "gtest/gtest.h"

#include <string_view>

#include "motis/elevators/parse_elevator_id_osm_mapping.h"

using namespace motis;
using namespace std::string_view_literals;

constexpr auto const kElevatorIdOsmMappingCsv = R"__(dhid,diid,osm_kind,osm_id
dbinfrago-temporary:d23340e4-ca1a-533e-803a-c036883147a3,diid:02b2be0f-c1da-1eef-a490-a02c488737ac,node,8891093860
de:01002:49320,diid:02b2be0f-c1da-1eef-a490-ddb6f99637ae,node,2505371425
de:01002:49320,diid:02b2be0f-c1da-1eef-a490-dfa6e17997ae,node,2505371422
de:01003:57774,diid:02b2be0f-c1da-1eef-a490-aec6aa7b37ad,node,2543654133
de:01004:66023,diid:02b2be0f-c1da-1eef-a490-a8aaa8ac17ac,node,3833491147
)__"sv;

TEST(motis, parse_elevator_id_osm_mapping) {
  auto const map = parse_elevator_id_osm_mapping(kElevatorIdOsmMappingCsv);

  ASSERT_EQ(5, map.size());
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-a02c488737ac", map.at(8891093860ULL));
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-ddb6f99637ae", map.at(2505371425ULL));
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-dfa6e17997ae", map.at(2505371422ULL));
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-aec6aa7b37ad", map.at(2543654133ULL));
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-a8aaa8ac17ac", map.at(3833491147ULL));
}
