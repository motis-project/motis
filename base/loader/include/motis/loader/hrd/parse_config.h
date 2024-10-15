#pragma once

#include <climits>
#include <limits>
#include <vector>

#include "boost/filesystem/path.hpp"

#include "utl/parser/cstr.h"

#ifdef HAVE_DESIGNATED_INITIALIZERS
#define INIT(f, ...) f = __VA_ARGS__
#else
#define INIT(f, ...) __VA_ARGS__
#endif

namespace motis::loader::hrd {

struct range_parse_information {
  utl::field from_eva_or_idx_;
  utl::field to_eva_or_idx_;
  utl::field from_hhmm_or_idx_;
  utl::field to_hhmm_or_idx_;
};

enum filename_key {
  ATTRIBUTES,
  STATIONS,
  COORDINATES,
  BITFIELDS,
  TRACKS,
  INFOTEXT,
  BASIC_DATA,
  CATEGORIES,
  DIRECTIONS,
  PROVIDERS,
  THROUGH_SERVICES,
  MERGE_SPLIT_SERVICES,
  TIMEZONES,
  FOOTPATHS,
  FOOTPATHS_EXT,
  MIN_CT_FILE
};

static constexpr auto const ENCODING = "ISO8859-1";
static constexpr auto const SCHEDULE_DATA = "fahrten";
static constexpr auto const CORE_DATA = "stamm";

struct config {
  struct {
    utl::field code_;
    utl::field text_mul_spaces_;
    utl::field text_normal_;
  } att_;
  struct {
    utl::field index_;
    utl::field value_;
  } bf_;
  struct {
    utl::field code_;
    utl::field output_rule_;
    utl::field name_;
  } cat_;
  struct {
    utl::field eva_;
    utl::field text_;
  } dir_;
  struct {
    unsigned line_length_;
    utl::field bitfield_;
    utl::field key1_nr_;
    utl::field key1_admin_;
    utl::field key2_nr_;
    utl::field key2_admin_;
    utl::field eva_begin_;
    utl::field eva_end_;
  } merge_spl_;
  struct {
    struct {
      utl::field eva_;
    } meta_stations_;
    struct {
      utl::field from_;
      utl::field to_;
      utl::field duration_;
    } footpaths_;
  } meta_;
  struct {
    struct {
      utl::field eva_;
      utl::field name_;
    } names_;
    struct {
      utl::field eva_;
      utl::field lng_;
      utl::field lat_;
    } coords_;
  } st_;
  struct {
    utl::field bitfield_;
    utl::field key1_nr_;
    utl::field key1_admin_;
    utl::field key2_nr_;
    utl::field key2_admin_;
    utl::field eva_;
  } th_s_;
  struct {
    utl::field type1_eva_;
    utl::field type1_first_valid_eva_;
    utl::field type2_eva_;
    utl::field type2_dst_to_midnight_;
    utl::field type3_dst_to_midnight1_;
    utl::field type3_bitfield_idx1_;
    utl::field type3_bitfield_idx2_;
    utl::field type3_dst_to_midnight2_;
    utl::field type3_dst_to_midnight3_;
  } tz_;
  struct {
    utl::field prov_nr_;
  } track_;
  struct {
    unsigned min_line_length_;
    utl::field eva_num_;
    utl::field train_num_;
    utl::field train_admin_;
    utl::field track_name_;
    utl::field time_;
    utl::field bitfield_;
  } track_rul_;
  struct {
    utl::field att_eva_;
    utl::field att_code_;
    utl::field cat_;
    utl::field line_;
    utl::field traff_days_;
    utl::field dir_;
  } s_info_;

  range_parse_information attribute_parse_info_;
  range_parse_information line_parse_info_;
  range_parse_information category_parse_info_;
  range_parse_information traffic_days_parse_info_;
  range_parse_information direction_parse_info_;

  utl::cstr version_;
  std::vector<std::vector<std::string>> required_files_;

  boost::filesystem::path core_data_;
  boost::filesystem::path fplan_;
  std::string fplan_file_extension_;

  bool convert_utf8_;

  const char* files(filename_key k, int index = 0) const {
    return required_files_[k][index].c_str();
  }

  bool is_available(filename_key const k) const {
    return !required_files_[k].empty();
  }
};

const config hrd_5_00_8 = {
    INIT(.att_,
         {
             INIT(.code_, {0, 2}),
             INIT(.text_mul_spaces_, {21, utl::field::MAX_SIZE}),
             INIT(.text_normal_, {12, utl::field::MAX_SIZE}),
         }),
    INIT(.bf_,
         {
             INIT(.index_, {0, 6}),
             INIT(.value_, {7, utl::field::MAX_SIZE}),
         }),
    INIT(.cat_,
         {
             INIT(.code_, {0, 3}),
             INIT(.output_rule_, {9, 2}),
             INIT(.name_, {12, 8}),
         }),
    INIT(.dir_,
         {
             INIT(.eva_, {0, 7}),
             INIT(.text_, {8, utl::field::MAX_SIZE}),
         }),
    INIT(.merge_spl_,
         {
             INIT(.line_length_, 53),
             INIT(.bitfield_, {47, 6}),
             INIT(.key1_nr_, {18, 5}),
             INIT(.key1_admin_, {25, 6}),
             INIT(.key2_nr_, {33, 5}),
             INIT(.key2_admin_, {40, 6}),
             INIT(.eva_begin_, {0, 7}),
             INIT(.eva_end_, {9, 7}),
         }),
    INIT(.meta_,
         {
             INIT(.meta_stations_, {INIT(.eva_, {0, 7})}),
             INIT(.footpaths_,
                  {
                      INIT(.from_, {0, 7}),
                      INIT(.to_, {8, 7}),
                      INIT(.duration_, {16, 3}),
                  }),
         }),
    INIT(.st_,
         {
             INIT(.names_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.name_, {12, utl::field::MAX_SIZE}),
                  }),
             INIT(.coords_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.lng_, {8, 10}),
                      INIT(.lat_, {19, 10}),
                  }),
         }),
    INIT(.th_s_,
         {
             INIT(.bitfield_, {34, 6}),
             INIT(.key1_nr_, {0, 5}),
             INIT(.key1_admin_, {6, 6}),
             INIT(.key2_nr_, {21, 5}),
             INIT(.key2_admin_, {27, 6}),
             INIT(.eva_, {13, 7}),
         }),
    INIT(.tz_,
         {
             INIT(.type1_eva_, {0, 7}),
             INIT(.type1_first_valid_eva_, {8, 7}),
             INIT(.type2_eva_, {0, 7}),
             INIT(.type2_dst_to_midnight_, {8, 5}),
             INIT(.type3_dst_to_midnight1_, {14, 5}),
             INIT(.type3_bitfield_idx1_, {20, 8}),
             INIT(.type3_bitfield_idx2_, {34, 8}),
             INIT(.type3_dst_to_midnight2_, {29, 4}),
             INIT(.type3_dst_to_midnight3_, {43, 4}),
         }),
    INIT(.track_, {INIT(.prov_nr_, {0, 5})}),
    INIT(.track_rul_,
         {
             INIT(.min_line_length_, 22),
             INIT(.eva_num_, {0, 7}),
             INIT(.train_num_, {8, 5}),
             INIT(.train_admin_, {14, 6}),
             INIT(.track_name_, {21, 8}),
             INIT(.time_, {30, 4}),
             INIT(.bitfield_, {35, 6}),
         }),
    INIT(.s_info_,
         {
             INIT(.att_eva_, {22, 6}),
             INIT(.att_code_, {3, 2}),
             INIT(.cat_, {3, 3}),
             INIT(.line_, {3, 5}),
             INIT(.traff_days_, {22, 6}),
             INIT(.dir_, {5, 7}),
         }),
    INIT(.attribute_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.line_parse_info_,
         {
             INIT(.from_eva_or_idx_, {9, 7}),
             INIT(.to_eva_or_idx_, {17, 7}),
             INIT(.from_hhmm_or_idx_, {25, 6}),
             INIT(.to_hhmm_or_idx_, {32, 6}),
         }),
    INIT(.category_parse_info_,
         {
             INIT(.from_eva_or_idx_, {7, 7}),
             INIT(.to_eva_or_idx_, {15, 7}),
             INIT(.from_hhmm_or_idx_, {23, 6}),
             INIT(.to_hhmm_or_idx_, {30, 6}),
         }),
    INIT(.traffic_days_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.direction_parse_info_,
         {
             INIT(.from_eva_or_idx_, {13, 7}),
             INIT(.to_eva_or_idx_, {21, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.version_, "hrd_5_00_8"),
    INIT(.required_files_, {{"attributd_int_int.101", "attributd_int.101"},
                            {"bahnhof.101"},
                            {"dbkoord_geo.101"},
                            {"bitfield.101"},
                            {"gleise.101"},
                            {"infotext.101"},
                            {"eckdaten.101"},
                            {"zugart_int.101"},
                            {"richtung.101"},
                            {"unternehmen_ris.101"},
                            {"durchbi.101"},
                            {"vereinig_vt.101"},
                            {"zeitvs.101"},
                            {"metabhf.101"},
                            {"metabhf_zusatz.101"},
                            {"minct.csv"}}),
    INIT(.core_data_, "stamm"),
    INIT(.fplan_, "fahrten"),
    INIT(.fplan_file_extension_, ""),
    INIT(.convert_utf8_, false),
};

const config hrd_5_20_26 = {
    INIT(.att_,
         {
             INIT(.code_, {0, 2}),
             INIT(.text_mul_spaces_, {21, utl::field::MAX_SIZE}),
             INIT(.text_normal_, {12, utl::field::MAX_SIZE}),
         }),
    INIT(.bf_,
         {
             INIT(.index_, {0, 6}),
             INIT(.value_, {7, utl::field::MAX_SIZE}),
         }),
    INIT(.cat_,
         {
             INIT(.code_, {0, 3}),
             INIT(.output_rule_, {9, 2}),
             INIT(.name_, {12, 8}),
         }),
    INIT(.dir_,
         {
             INIT(.eva_, {0, 7}),
             INIT(.text_, {8, utl::field::MAX_SIZE}),
         }),
    INIT(.merge_spl_,
         {
             INIT(.line_length_, 50),
             INIT(.bitfield_, {44, 6}),
             INIT(.key1_nr_, {16, 6}),
             INIT(.key1_admin_, {23, 6}),
             INIT(.key2_nr_, {30, 6}),
             INIT(.key2_admin_, {37, 6}),
             INIT(.eva_begin_, {0, 7}),
             INIT(.eva_end_, {8, 7}),
         }),
    INIT(.meta_,
         {
             INIT(.meta_stations_, {INIT(.eva_, {0, 7})}),
             INIT(.footpaths_,
                  {
                      INIT(.from_, {0, 7}),
                      INIT(.to_, {8, 7}),
                      INIT(.duration_, {16, 3}),
                  }),
         }),
    INIT(.st_,
         {
             INIT(.names_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.name_, {12, utl::field::MAX_SIZE}),
                  }),
             INIT(.coords_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.lng_, {8, 10}),
                      INIT(.lat_, {19, 10}),
                  }),
         }),
    INIT(.th_s_,
         {
             INIT(.bitfield_, {34, 6}),
             INIT(.key1_nr_, {0, 5}),
             INIT(.key1_admin_, {6, 6}),
             INIT(.key2_nr_, {21, 5}),
             INIT(.key2_admin_, {27, 6}),
             INIT(.eva_, {13, 7}),
         }),
    INIT(.tz_,
         {
             INIT(.type1_eva_, {0, 7}),
             INIT(.type1_first_valid_eva_, {8, 7}),
             INIT(.type2_eva_, {0, 7}),
             INIT(.type2_dst_to_midnight_, {8, 5}),
             INIT(.type3_dst_to_midnight1_, {14, 5}),
             INIT(.type3_bitfield_idx1_, {20, 8}),
             INIT(.type3_bitfield_idx2_, {34, 8}),
             INIT(.type3_dst_to_midnight2_, {29, 4}),
             INIT(.type3_dst_to_midnight3_, {43, 4}),
         }),
    INIT(.track_, {INIT(.prov_nr_, {0, 5})}),
    INIT(.track_rul_,
         {
             INIT(.min_line_length_, 22),
             INIT(.eva_num_, {0, 7}),
             INIT(.train_num_, {8, 5}),
             INIT(.train_admin_, {14, 6}),
             INIT(.track_name_, {21, 8}),
             INIT(.time_, {30, 4}),
             INIT(.bitfield_, {35, 6}),
         }),
    INIT(.s_info_,
         {
             INIT(.att_eva_, {22, 6}),
             INIT(.att_code_, {3, 2}),
             INIT(.cat_, {3, 3}),
             INIT(.line_, {3, 8}),
             INIT(.traff_days_, {22, 6}),
             INIT(.dir_, {5, 7}),
         }),
    INIT(.attribute_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.line_parse_info_,
         {
             INIT(.from_eva_or_idx_, {12, 7}),
             INIT(.to_eva_or_idx_, {20, 7}),
             INIT(.from_hhmm_or_idx_, {28, 6}),
             INIT(.to_hhmm_or_idx_, {35, 6}),
         }),
    INIT(.category_parse_info_,
         {
             INIT(.from_eva_or_idx_, {7, 7}),
             INIT(.to_eva_or_idx_, {15, 7}),
             INIT(.from_hhmm_or_idx_, {23, 6}),
             INIT(.to_hhmm_or_idx_, {30, 6}),
         }),
    INIT(.traffic_days_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.direction_parse_info_,
         {
             INIT(.from_eva_or_idx_, {13, 7}),
             INIT(.to_eva_or_idx_, {21, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.version_, "hrd_5_20_26"),
    INIT(.required_files_, {{"attributd.txt"},
                            {"bahnhof.txt"},
                            {"bfkoord.txt"},
                            {"bitfield.txt"},
                            {"gleise.txt"},
                            {"infotext.txt"},
                            {"eckdaten.txt"},
                            {"zugart.txt"},
                            {"richtung.txt"},
                            {"unternehmen_ris.txt"},
                            {"durchbi.txt"},
                            {"vereinig_vt.txt"},
                            {"zeitvs.txt"},
                            {"metabhf.txt"},
                            {},
                            {"minct.csv"}}),
    INIT(.core_data_, "stamm"),
    INIT(.fplan_, "fahrten"),
    INIT(.fplan_file_extension_, ""),
    INIT(.convert_utf8_, false),
};

const config hrd_5_20_39 = {
    INIT(.att_,
         {
             INIT(.code_, {0, 2}),
             INIT(.text_mul_spaces_, {21, utl::field::MAX_SIZE}),
             INIT(.text_normal_, {12, utl::field::MAX_SIZE}),
         }),
    INIT(.bf_,
         {
             INIT(.index_, {0, 6}),
             INIT(.value_, {7, utl::field::MAX_SIZE}),
         }),
    INIT(.cat_,
         {
             INIT(.code_, {0, 3}),
             INIT(.output_rule_, {9, 1}),
             INIT(.name_, {11, 8}),
         }),
    INIT(.dir_,
         {
             INIT(.eva_, {0, 7}),
             INIT(.text_, {8, utl::field::MAX_SIZE}),
         }),
    INIT(.merge_spl_,
         {
             INIT(.line_length_, 50),
             INIT(.bitfield_, {44, 6}),
             INIT(.key1_nr_, {16, 6}),
             INIT(.key1_admin_, {23, 6}),
             INIT(.key2_nr_, {30, 6}),
             INIT(.key2_admin_, {37, 6}),
             INIT(.eva_begin_, {0, 7}),
             INIT(.eva_end_, {8, 7}),
         }),
    INIT(.meta_,
         {
             INIT(.meta_stations_, {INIT(.eva_, {0, 7})}),
             INIT(.footpaths_,
                  {
                      INIT(.from_, {0, 7}),
                      INIT(.to_, {8, 7}),
                      INIT(.duration_, {16, 3}),
                  }),
         }),
    INIT(.st_,
         {
             INIT(.names_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.name_, {12, utl::field::MAX_SIZE}),
                  }),
             INIT(.coords_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.lng_, {8, 10}),
                      INIT(.lat_, {19, 10}),
                  }),
         }),
    INIT(.th_s_,
         {
             INIT(.bitfield_, {34, 6}),
             INIT(.key1_nr_, {0, 5}),
             INIT(.key1_admin_, {6, 6}),
             INIT(.key2_nr_, {21, 5}),
             INIT(.key2_admin_, {27, 6}),
             INIT(.eva_, {13, 7}),
         }),
    INIT(.tz_,
         {
             INIT(.type1_eva_, {0, 7}),
             INIT(.type1_first_valid_eva_, {8, 7}),
             INIT(.type2_eva_, {0, 7}),
             INIT(.type2_dst_to_midnight_, {8, 5}),
             INIT(.type3_dst_to_midnight1_, {14, 5}),
             INIT(.type3_bitfield_idx1_, {20, 8}),
             INIT(.type3_bitfield_idx2_, {34, 8}),
             INIT(.type3_dst_to_midnight2_, {29, 4}),
             INIT(.type3_dst_to_midnight3_, {43, 4}),
         }),
    INIT(.track_, {INIT(.prov_nr_, {0, 5})}),
    INIT(.track_rul_,
         {
             INIT(.min_line_length_, 34),
             INIT(.eva_num_, {0, 7}),
             INIT(.train_num_, {8, 5}),
             INIT(.train_admin_, {14, 6}),
             INIT(.track_name_, {21, 8}),
             INIT(.time_, {30, 4}),
             INIT(.bitfield_, {35, 6}),
         }),
    INIT(.s_info_,
         {
             INIT(.att_eva_, {22, 6}),
             INIT(.att_code_, {3, 2}),
             INIT(.cat_, {3, 3}),
             INIT(.line_, {3, 8}),
             INIT(.traff_days_, {22, 6}),
             INIT(.dir_, {5, 7}),
         }),
    INIT(.attribute_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.line_parse_info_,
         {
             INIT(.from_eva_or_idx_, {12, 7}),
             INIT(.to_eva_or_idx_, {20, 7}),
             INIT(.from_hhmm_or_idx_, {28, 6}),
             INIT(.to_hhmm_or_idx_, {35, 6}),
         }),
    INIT(.category_parse_info_,
         {
             INIT(.from_eva_or_idx_, {7, 7}),
             INIT(.to_eva_or_idx_, {15, 7}),
             INIT(.from_hhmm_or_idx_, {23, 6}),
             INIT(.to_hhmm_or_idx_, {30, 6}),
         }),
    INIT(.traffic_days_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.direction_parse_info_,
         {
             INIT(.from_eva_or_idx_, {13, 7}),
             INIT(.to_eva_or_idx_, {21, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.version_, "hrd_5_20_39"),
    INIT(.required_files_, {{"ATTRIBUT_DE"},
                            {"BAHNHOF"},
                            {"BFKOORD_GEO", "BFKOORD_WGS"},
                            {"BITFELD"},
                            {"GLEIS"},
                            {"INFOTEXT_DE"},
                            {"ECKDATEN"},
                            {"ZUGART"},
                            {"RICHTUNG"},
                            {"BETRIEB_DE"},
                            {"DURCHBI"},
                            {/* vereinig */},
                            {"ZEITVS"},
                            {"METABHF"},
                            {/* meta bhf zusatz */},
                            {/* min ct csv */}}),
    INIT(.core_data_, "."),
    INIT(.fplan_, "FPLAN"),
    INIT(.fplan_file_extension_, ""),
    INIT(.convert_utf8_, true),
};

const config hrd_5_20_avv = {
    INIT(.att_,
         {
             INIT(.code_, {0, 2}),
             INIT(.text_mul_spaces_, {21, utl::field::MAX_SIZE}),
             INIT(.text_normal_, {12, utl::field::MAX_SIZE}),
         }),
    INIT(.bf_,
         {
             INIT(.index_, {0, 6}),
             INIT(.value_, {7, utl::field::MAX_SIZE}),
         }),
    INIT(.cat_,
         {
             INIT(.code_, {0, 3}),
             INIT(.output_rule_, {9, 1}),
             INIT(.name_, {11, 8}),
         }),
    INIT(.dir_,
         {
             INIT(.eva_, {0, 7}),
             INIT(.text_, {8, utl::field::MAX_SIZE}),
         }),
    INIT(.merge_spl_,
         {
             INIT(.line_length_, 50),
             INIT(.bitfield_, {std::numeric_limits<size_t>::max(), 0U}),
             INIT(.key1_nr_, {16, 5}),
             INIT(.key1_admin_, {22, 6}),
             INIT(.key2_nr_, {29, 5}),
             INIT(.key2_admin_, {35, 6}),
             INIT(.eva_begin_, {0, 7}),
             INIT(.eva_end_, {8, 7}),
         }),
    INIT(.meta_,
         {
             INIT(.meta_stations_, {INIT(.eva_, {0, 7})}),
             INIT(.footpaths_,
                  {
                      INIT(.from_, {0, 7}),
                      INIT(.to_, {8, 7}),
                      INIT(.duration_, {16, 3}),
                  }),
         }),
    INIT(.st_,
         {
             INIT(.names_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.name_, {12, utl::field::MAX_SIZE}),
                  }),
             INIT(.coords_,
                  {
                      INIT(.eva_, {0, 7}),
                      INIT(.lng_, {8, 10}),
                      INIT(.lat_, {19, 10}),
                  }),
         }),
    INIT(.th_s_,
         {
             INIT(.bitfield_, {34, 6}),
             INIT(.key1_nr_, {0, 5}),
             INIT(.key1_admin_, {6, 6}),
             INIT(.key2_nr_, {21, 5}),
             INIT(.key2_admin_, {27, 6}),
             INIT(.eva_, {13, 7}),
         }),
    INIT(.tz_,
         {
             INIT(.type1_eva_, {0, 7}),
             INIT(.type1_first_valid_eva_, {8, 7}),
             INIT(.type2_eva_, {0, 7}),
             INIT(.type2_dst_to_midnight_, {8, 5}),
             INIT(.type3_dst_to_midnight1_, {14, 5}),
             INIT(.type3_bitfield_idx1_, {20, 8}),
             INIT(.type3_bitfield_idx2_, {34, 8}),
             INIT(.type3_dst_to_midnight2_, {29, 4}),
             INIT(.type3_dst_to_midnight3_, {43, 4}),
         }),
    INIT(.track_, {INIT(.prov_nr_, {0, 5})}),
    INIT(.track_rul_,
         {
             INIT(.min_line_length_, 22),
             INIT(.eva_num_, {0, 7}),
             INIT(.train_num_, {8, 5}),
             INIT(.train_admin_, {14, 6}),
             INIT(.track_name_, {21, 8}),
             INIT(.time_, {30, 4}),
             INIT(.bitfield_, {35, 6}),
         }),
    INIT(.s_info_,
         {
             INIT(.att_eva_, {22, 6}),
             INIT(.att_code_, {3, 2}),
             INIT(.cat_, {3, 3}),
             INIT(.line_, {3, 8}),
             INIT(.traff_days_, {22, 6}),
             INIT(.dir_, {5, 7}),
         }),
    INIT(.attribute_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.line_parse_info_,
         {
             INIT(.from_eva_or_idx_, {12, 7}),
             INIT(.to_eva_or_idx_, {20, 7}),
             INIT(.from_hhmm_or_idx_, {28, 6}),
             INIT(.to_hhmm_or_idx_, {35, 6}),
         }),
    INIT(.category_parse_info_,
         {
             INIT(.from_eva_or_idx_, {7, 7}),
             INIT(.to_eva_or_idx_, {15, 7}),
             INIT(.from_hhmm_or_idx_, {23, 6}),
             INIT(.to_hhmm_or_idx_, {30, 6}),
         }),
    INIT(.traffic_days_parse_info_,
         {
             INIT(.from_eva_or_idx_, {6, 7}),
             INIT(.to_eva_or_idx_, {14, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.direction_parse_info_,
         {
             INIT(.from_eva_or_idx_, {13, 7}),
             INIT(.to_eva_or_idx_, {21, 7}),
             INIT(.from_hhmm_or_idx_, {29, 6}),
             INIT(.to_hhmm_or_idx_, {36, 6}),
         }),
    INIT(.version_, "hrd_5_20_avv"),
    INIT(.required_files_, {{"attribut_de"},
                            {"bahnhof"},
                            {"bfkoord"},
                            {"bitfeld"},
                            {"gleise"},
                            {"infotext"},
                            {"eckdaten"},
                            {"zugart"},
                            {"richtung"},
                            {"betrieb"},
                            {"durchbi"},
                            {"vereinig"},
                            {"zeitvs"},
                            {"metabf"},
                            {/* meta bhf zusatz */},
                            {/* min ct csv */}}),
    INIT(.core_data_, "."),
    INIT(.fplan_, "."),
    INIT(.fplan_file_extension_, ".LIN"),
    INIT(.convert_utf8_, false),
};

const std::vector<config> configs = {hrd_5_00_8, hrd_5_20_26, hrd_5_20_39,
                                     hrd_5_20_avv};

}  // namespace motis::loader::hrd
