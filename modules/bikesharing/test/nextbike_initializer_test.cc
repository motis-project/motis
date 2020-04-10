#include "gtest/gtest.h"

#include <iostream>

#include "boost/date_time/posix_time/posix_time.hpp"

#include "boost/date_time/c_local_time_adjustor.hpp"
#include "boost/date_time/local_time_adjustor.hpp"

#include "utl/parser/buffer.h"

#include "motis/bikesharing/nextbike_initializer.h"
#include "motis/bikesharing/terminal.h"

namespace motis::bikesharing {

char const* xml_fixture = R"((
<?xml version="1.0" encoding="utf-8"?>
<markers>
  <country lat="50.7086" lng="10.6348" zoom="5" name="nextbike Germany"
      hotline="+493069205046" domain="de" country="DE" country_name="Germany"
      terms="http://www.nextbike.de/media/nextbike_AGB_de_en_03-2015.pdf"
      website="http://www.nextbike.de/">
    <city uid="1" lat="51.3415" lng="12.3625" zoom="14" maps_icon=""
        alias="leipzig" break="0" name="Leipzig">
      <place uid="28" lat="51.3405051597014" lng="12.3688137531281"
          name="Gottschedstraße/Bosestraße " spot="1" number="4013" bikes="3"
          terminal_type="unknown" bike_numbers="10042,10657,10512" />
      <place uid="128" lat="51.3371237726003" lng="12.37330377101898"
          name="Burgplatz/Freifläche/Zaun" spot="1" number="4011" bikes="5+"
          terminal_type="unknown" bike_numbers="10520,10452,10114,10349,10297" />
    </city>
  </country>
  <!-- only germany to simplify timezone handling -->
  <country lat="35.066" lng="33.3984" zoom="8" name="nextbike Cyprus"
      hotline="+35777788323" domain="cy" country="CY" country_name="Cyprus"
      terms="http://www.nextbike.com.cy/uploads/media/Oroi_kai_Proupotheseis_Xrhshs_Nextbike_Cy_Ltd.pdf"
      website="http://www.nextbike.com.cy/">
    <city uid="190" lat="34.6823" lng="33.0464" zoom="13" maps_icon=""
        alias="" break="0" name="Limassol">
      <place uid="115696" lat="34.67127772425186" lng="33.04337739944458"
          name="Molos" spot="1" number="6750" bikes="5+" bike_racks="15"
          bike_numbers="67610,67558,67626,67661,67510" />
    </city>
  </country>
</markers>
))";

TEST(bikesharing_nextbike_initializer, parser_test) {
  auto result = nextbike_parse_xml(utl::buffer{xml_fixture});

  ASSERT_EQ(2, result.size());

  auto r0 = result[0];
  EXPECT_EQ(std::string{"28"}, r0.uid_);
  EXPECT_EQ(51.3405051597014, r0.lat_);
  EXPECT_EQ(12.3688137531281, r0.lng_);
  EXPECT_EQ(std::string{"Gottschedstraße/Bosestraße "}, r0.name_);
  EXPECT_EQ(3, r0.available_bikes_);

  auto r1 = result[1];
  EXPECT_EQ(std::string{"128"}, r1.uid_);
  EXPECT_EQ(51.3371237726003, r1.lat_);
  EXPECT_EQ(12.37330377101898, r1.lng_);
  EXPECT_EQ(std::string{"Burgplatz/Freifläche/Zaun"}, r1.name_);
  EXPECT_EQ(5, r1.available_bikes_);
}

TEST(bikesharing_nextbike_initializer, filename_to_timestamp) {
  std::string filename{"nextbike-1432133475.xml"};
  EXPECT_EQ(1432133475, nextbike_filename_to_timestamp(filename));
}

}  // namespace motis::bikesharing
