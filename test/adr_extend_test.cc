#include "gtest/gtest.h"

#include "adr/score.h"

#include "motis/adr_extend_tt.h"

namespace motis {

adr::score_t get_diff(std::string, std::string, std::vector<adr::sift_offset>&);

}

using namespace motis;

TEST(adr_extend, str_diff_test) {
  auto sift4_dist = std::vector<adr::sift_offset>{};
  // auto const str_match_score =
  //     get_diff("Fribourg, Route-de-Tavel", "Fribourg, Bellevue", sift4_dist);
  // std::cout << str_match_score << "\n";
  //
  // auto const str_match_score1 =
  //     get_diff("Bern", "Bern, Hauptbahnhof", sift4_dist);
  // std::cout << str_match_score1 << "\n";
  //
  // auto const str_match_score2 =
  //     get_diff("Kiel Hbf/Kaistraße", "Kiel Bahnhof (Fähre)", sift4_dist);
  // std::cout << str_match_score2 << "\n";
  // auto const str_match_score3 =
  //     get_diff("Kiel Bahnhof (Fähre)", "Kiel Hbf/Kaistraße", sift4_dist);
  // std::cout << str_match_score3 << "\n";

  auto const str_match_score4 = get_diff("Halle (Saale), Hauptbahnhof/ZOB",
                                         "Halle(Saale)Hbf", sift4_dist);
  std::cout << str_match_score4 << "\n";
}