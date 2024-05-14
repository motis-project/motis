#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/timetable.h"
#include "icc/match.h"
#include "osr/extract/extract.h"
#include "osr/platforms.h"
#include "osr/ways.h"

namespace fs = std::filesystem;

constexpr auto const gtfs = R"(
# stops.txt
stop_name,parent_station,stop_id,stop_desc,stop_lat,stop_lon,location_type,wheelchair_boarding,level_id,platform_code
Frankfurt (Main) Hauptbahnhof,,de:06412:10,,50.10681,8.662657,1,,,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:1:32,Busbahnhof Linie 37,50.106537,8.664588,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:24:40,Busbahnhof Linie 33 und 35,50.10662,8.664588,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:2:41,Busbahnhof Linie 46,50.10668,8.664503,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:3:33,Vor Hauptportal NBaseler Platz | Nachtbus VPlatz der Repu,50.107304,8.664751,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:4:34,Nachtbus NPDR,50.107815,8.664943,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:7:37,Strab NPLDR,50.107735,8.664706,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:8:38,Strab VPLDR,50.107616,8.664678,,,2,
Frankfurt (Main) Hauptbahnhof,,de:06412:10:9:39,Vor Busbahnhof NHbf Südseite/Pforzh.,50.10637,8.664869,,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001027,Zugang M,50.106945,8.665494,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001028,Zugang Mu,50.1068,8.665425,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001029,Zugang N,50.108387,8.66389,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001030,Zugang Nu,50.1083,8.664031,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001032,Busbahnhof Linie 37,50.106537,8.664588,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001033,Vor Hauptportal NBaseler Platz | Nachtbus VPlatz der Repu,50.107304,8.664751,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001034,Nachtbus NPDR,50.107815,8.664943,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001037,Strab NPLDR,50.107735,8.664706,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001038,Strab VPLDR,50.107616,8.664678,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001039,Vor Busbahnhof NHbf Südseite/Pforzh.,50.10637,8.664869,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001040,Busbahnhof Linie 33 und 35,50.10662,8.664588,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001041,Busbahnhof Linie 46,50.10668,8.664503,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001042,Zugang Ou,50.108414,8.663555,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001044,Zugang S,50.10673,8.665663,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001045,Zugang Su,50.106674,8.665454,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001046,Zugang T,50.106487,8.665455,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001047,Zugang Tu,50.106613,8.665426,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001048,Z+1A,50.10602,8.663865,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001049,Zugang G,50.10635,8.664883,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001051,V+1Ao,50.107136,8.662948,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001052,V+1Au,50.10725,8.662766,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001053,V+1Bo,50.107502,8.66261,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001054,V+1Bu,50.10735,8.662681,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001055,V+1Co,50.107506,8.663128,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001056,V+1Cu,50.10755,8.663253,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001057,V+1Do,50.107147,8.663801,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001058,V+1Du,50.107227,8.663996,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001059,V+1Eo,50.10728,8.662919,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001060,Zugang U,50.107693,8.665349,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001061,V+1Eu,50.10728,8.662919,3,,8,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001062,V+1Fo,50.10742,8.662793,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001063,Zugang Uu,50.10761,8.665154,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001064,V+1Fu,50.10742,8.662793,3,,8,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001065,V+1Go,50.10745,8.662401,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001066,V+1Gu,50.107456,8.662401,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001067,Übergangspunkt D,50.107647,8.663113,2,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001068,Zugang Bo,50.10798,8.662341,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001069,Zugang Ao,50.105907,8.664271,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001070,Zugang Hu,50.10716,8.664542,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001071,V-1Ao,50.108376,8.663066,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001072,V-1Au,50.108322,8.663178,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001073,V-1Bo,50.108276,8.66294,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001074,V-1Bu,50.108215,8.663025,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001075,V-1Co,50.108036,8.66332,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001076,V-1Cu,50.108025,8.66332,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001077,V-1Do,50.108154,8.663486,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001078,V-1Du,50.108135,8.663487,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001079,V-1Eo,50.107777,8.663461,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001080,V-1Eu,50.10763,8.663169,3,,8,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001085,V-1Io,50.107624,8.664385,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001086,V-1Iu,50.10769,8.664287,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001087,V-1Mo,50.10763,8.663588,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001088,V-1Mu,50.107605,8.663518,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001089,V-1No,50.10772,8.663489,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001090,V-1Nu,50.107704,8.66342,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001091,Z-2A,50.10763,8.663434,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001092,V-2Eo,50.10765,8.663476,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001093,V-2Eu,50.107677,8.663546,3,,3,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001094,Z-3C,50.107685,8.663588,3,,3,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001095,V-3Co,50.107765,8.663475,3,,3,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001096,V-3Cu,50.107704,8.663336,3,,8,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001097,V-3Do,50.107605,8.663686,3,,3,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001098,V-3Du,50.10754,8.663547,3,,8,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001099,Z-1E,50.107277,8.662682,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001106,Übergangspunkt P,50.108147,8.664367,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001107,Übergangspunkt O,50.108475,8.663386,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001108,Übergangspunkt B,50.10798,8.662453,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001109,Übergangspunkt N,50.108368,8.663806,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001111,Übergangspunkt L,50.107788,8.664328,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001112,Übergangspunkt M,50.107025,8.665521,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001113,Übergangspunkt K,50.107735,8.664971,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001114,Übergangspunkt J,50.107304,8.664904,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001115,Übergangspunkt I,50.107147,8.664095,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001116,Übergangspunkt H,50.106724,8.66407,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001117,Zugang A,50.105927,8.664215,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001118,Übergangspunkt G,50.106293,8.664324,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001119,Z-1WW,50.108116,8.663221,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001121,Übergangspunkt U,50.107555,8.665196,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001122,Übergangspunkt T,50.10638,8.665498,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001123,Übergangspunkt S,50.10655,8.665539,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001124,Zugang Ju,50.107292,8.664946,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001125,Zugang K,50.107628,8.664916,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001126,Zugang L,50.107883,8.664313,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001127,Übergangspunkt A,50.10589,8.664131,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001129,Zugang J,50.107304,8.664904,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001170,Zugang R,50.108364,8.662548,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001171,Zugang Pu,50.1082,8.664185,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001174,Zugang Qu,50.108025,8.662998,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001250,Zugang F,50.105362,8.662569,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001251,V-1Fo,50.107605,8.663658,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001252,V-1Fu,50.10746,8.663365,3,,8,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001253,V-1Ho,50.107464,8.664134,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001254,V-1Hu,50.107517,8.66405,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001256,Zugang Iu,50.107357,8.664666,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001257,Zugang D,50.10758,8.663365,2,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001258,Zugang O,50.10846,8.663386,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001259,Zugang Ru,50.108284,8.662703,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001260,Zugang I,50.107285,8.664485,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001262,Z0A,50.106052,8.664578,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001263,Z0B,50.107292,8.664247,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001264,Z0C,50.10777,8.664845,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001265,Z0D,50.107735,8.665307,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001266,Z0E,50.10691,8.665774,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001267,Z0F,50.106632,8.665706,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001268,Z0G,50.1066,8.66442,3,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001271,Zugang P,50.1083,8.664366,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001272,Zugang Gu,50.106415,8.66505,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001273,Zugang H,50.107178,8.664556,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001274,Übergangspunkt E,50.10714,8.663801,2,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001275,Zugang E,50.107254,8.664052,2,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001276,Übergangspunkt C,50.107937,8.662887,2,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001277,Zugang B,50.10797,8.662313,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001278,Zugang Ku,50.107655,8.664748,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001279,Zugang Lu,50.10801,8.664158,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001280,Z+1B,50.106106,8.663794,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001281,Z+1C,50.106274,8.663639,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001282,Z+1D,50.1064,8.663527,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001283,Z+1E,50.106533,8.6634,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001284,Z+1F,50.106686,8.663245,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001285,Z+1G,50.10684,8.663104,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001286,Z+1H,50.10699,8.662963,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001287,Z+1I,50.107143,8.662823,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001288,Z+1J,50.107277,8.662696,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001289,Z+1K,50.10742,8.662555,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001290,Z+1L,50.107494,8.662485,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001291,Z+1M,50.107582,8.6624,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001292,Z+1N,50.107655,8.66233,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001294,Z+1P,50.10721,8.662766,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001295,Z+1Q,50.10735,8.662625,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001296,Z+1R,50.10718,8.663731,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001297,Z+1S,50.1071,8.663815,3,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001350,Z-1A,50.10663,8.665272,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001351,Z-1B,50.106735,8.665174,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001352,Z-1C,50.107258,8.664695,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001354,Z-1D,50.1075,8.664791,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001355,Z-1E2,50.107418,8.664596,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001356,Z-1F,50.107327,8.664401,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001357,Z-1G,50.107338,8.664261,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001358,Z-1H,50.10765,8.663728,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001359,Z-1I,50.107685,8.663671,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001360,Z-1J,50.107723,8.663615,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001361,Z-1K,50.107765,8.663559,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001362,Z-1L,50.107803,8.663559,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001363,Z-1M,50.107803,8.663503,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001364,Z-1N,50.107758,8.663769,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001365,Z-1O,50.107796,8.663727,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001366,Z-1P,50.107838,8.663656,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001367,Z-1Q,50.107876,8.6636,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001368,Z-1R,50.108093,8.66406,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001369,Z-1S,50.10811,8.663361,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001370,Z-1T,50.108356,8.662954,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001371,Z-1U,50.10819,8.663416,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001372,Z-1W,50.108135,8.663333,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001375,Z-1Y,50.107517,8.664847,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001376,ÜV-2Ao,50.107758,8.663727,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001377,ÜV-2Bo,50.107857,8.663978,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001378,ÜV-2Do,50.108,8.663753,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001379,ÜV-2Co,50.107918,8.663488,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001403,Übergangspunkt F,50.105408,8.662639,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001404,Übergangspunkt R,50.10835,8.66259,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001405,Übergabepunkt Q,50.108017,8.663166,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001468,Zugang C,50.1079,8.662677,2,,5,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001469,Zugang Q,50.107952,8.662831,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001470,Zugang Ku2,50.107662,8.664748,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001471,Zugang K2,50.107628,8.664958,2,,2,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001472,V-1Ho2,50.10747,8.664148,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001473,V-1Hu2,50.107525,8.664064,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001474,V-1Bu2,50.108223,8.663039,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001475,V-1Bo2,50.10827,8.662926,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001476,V-1Au2,50.108315,8.663178,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001477,V-1Ao2,50.108383,8.663079,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001478,V-1Io2,50.107635,8.664399,3,,1,
Frankfurt (Main) Hauptbahnhof,de:06412:10,000320001479,V-1Iu2,50.107677,8.664273,3,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:10:15,Gleis 1,50.10577,8.663461,,,5,1
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:11:8,Gleis 1a,50.104183,8.659307,,,5,1a
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:12:10,Gleis 3,50.10587,8.663474,,,5,3
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:12:13,Gleis 2,50.105843,8.663489,,,5,2
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:13:6,Gleis 5,50.106003,8.663306,,,5,5
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:13:7,Gleis 4,50.10597,8.663334,,,5,4
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:14:4,Gleis 7,50.106148,8.663123,,,5,7
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:14:5,Gleis 6,50.106102,8.663151,,,5,6
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:15:2,Gleis 9,50.106262,8.662941,,,5,9
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:15:3,Gleis 8,50.1062,8.663011,,,5,8
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:16:12,Gleis 10,50.1063,8.662814,,,5,10
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:16:16,Gleis 11,50.10636,8.662772,,,5,11
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:17:18,Gleis 12,50.10647,8.662687,,,5,12
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:17:20,Gleis 13,50.10652,8.662645,,,5,13
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:18:22,Gleis 14,50.10662,8.662561,,,5,14
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:18:23,Gleis 15,50.106686,8.662504,,,5,15
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:19:24,Gleis 16,50.106754,8.662378,,,5,16
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:19:25,Gleis 17,50.10681,8.66235,,,5,17
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:20:1,Gleis 19,50.10696,8.662195,,,5,19
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:20:26,Gleis 18,50.1069,8.662237,,,5,18
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:21:14,Gleis 21,50.107098,8.662068,,,5,21
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:21:9,Gleis 20,50.107033,8.662124,,,5,20
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:22:17,Gleis 22,50.10716,8.661984,,,5,22
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:22:19,Gleis 23,50.107224,8.661913,,,5,23
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:23:21,Gleis 24,50.107323,8.661773,,,5,24
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:5:35,NMesse/Ausst.,50.107903,8.663824,,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:6:31,NW.-Brandt-Platz,50.107765,8.663615,,,4,
Frankfurt (Main) Hauptbahnhof,de:06412:10,de:06412:10:6:36,NW.-Brandt-Platz,50.10783,8.663698,,,4,
Frankfurt (Main) Hauptbahnhof Südseite,,de:06412:7011,,50.105064,8.662115,,,,
Frankfurt (Main) Hauptbahnhof Südseite,,de:06412:7011:31:31,NHeilbro,50.10561,8.6637,,,2,
Frankfurt (Main) Hauptbahnhof Südseite,,de:06412:7011:32:32,VHeilbronner,50.105656,8.664049,,,2,
Frankfurt (Main) Hauptbahnhof tief,,de:06412:7010,,50.107162,8.662501,1,,,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701003,Z-1E,50.107224,8.662556,3,,1,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701004,V-1Go,50.107117,8.662571,3,,1,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701005,V-1Gu,50.10701,8.662334,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701006,V-1Jo,50.107304,8.662416,3,,1,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701007,V-1Ju,50.107197,8.662165,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701010,ÜV-1FU,50.10746,8.663379,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701011,V-4Au,50.10725,8.662948,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701012,V-4Ao,50.107124,8.662655,3,,1,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701013,V-4Bu,50.107456,8.66275,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701014,V-4Bo,50.10734,8.662486,3,,1,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701015,Z-3A,50.107895,8.663726,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701016,V-2Co,50.107937,8.663488,3,,4,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701017,V-2Cu,50.107883,8.663586,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701018,V-2Do,50.108063,8.663697,3,,4,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701019,V-2Du,50.108,8.663795,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701020,Z-3B,50.107704,8.663951,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701021,V-2Ao,50.107777,8.663727,3,,4,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701022,V-2Au,50.107697,8.663825,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701023,V-2Bo,50.107903,8.66395,3,,4,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701024,V-2Bu,50.107822,8.664048,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701025,Z-3D,50.10642,8.660591,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701026,V-4Cu,50.106518,8.660632,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701027,V-4Co,50.10649,8.660521,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701028,V-4Du,50.106373,8.660759,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701029,V-4Do,50.106354,8.660647,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701030,ÜV-1Eu,50.10766,8.663182,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701031,V-3Ao,50.107586,8.663742,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701032,V-3Au,50.10757,8.6637,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701033,V-3Bo,50.10779,8.663531,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701034,V-3Bu,50.107765,8.663489,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701035,ÜV+1Eu,50.107296,8.662947,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701036,ÜV+1Fu,50.10745,8.662806,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701037,ÜV-3Du,50.10754,8.663547,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701038,ÜV-3Cu,50.10772,8.663336,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701039,V-3Eu,50.106926,8.662139,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701040,V-3Fu,50.107117,8.66197,3,,8,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701041,V-3Eo,50.106953,8.662237,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701042,V-3Fo,50.107132,8.662068,3,,3,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,000320701099,Z,50.107273,8.664359,2,,2,
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,de:06412:7010:1:3,Schiene S-Bahn Gleis 102,50.107136,8.662473,,,8,102
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,de:06412:7010:1:4,Schiene S-Bahn Gleis 101,50.107136,8.662473,,,8,101
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,de:06412:7010:2:1,Schiene S-Bahn Gleis 104,50.107136,8.662473,,,8,104
Frankfurt (Main) Hauptbahnhof tief,de:06412:7010,de:06412:7010:2:2,Schiene S-Bahn Gleis 103,50.107136,8.662473,,,8,103
Frankfurt (Main) Hauptbahnhof/Fernbusterminal,,de:06412:60865:1:1,Ffm Hauptbahnhof/Fernbusterminal | Stuttgarter Str.,50.10454,8.66182,,,2,Q
Frankfurt (Main) Hauptbahnhof/Fernbusterminal,,de:06412:60865:1:2,Ausstieg/Warten,50.104362,8.661989,,,2,R
Frankfurt (Main) Hauptbahnhof/Fernbusterminal,,de:06412:60865:2:3,Ausstieg/Warten/Einstieg,50.10515,8.663368,,,2,T
Frankfurt (Main) Hauptbahnhof/Fernbusterminal,,de:06412:60865:3:4,NBeh.zentr,50.105118,8.66204,,,2,P
Frankfurt (Main) Hauptbahnhof/Fernbusterminal,,de:06412:60865:4:5,AusstiegWarten NHbf,50.104557,8.663386,,,2,
Frankfurt (Main) Hauptbahnhof/Karlstraße,,de:06412:60767:1:1,NMainzer Landstra,50.109142,8.663605,,,2,
Frankfurt (Main) Hauptbahnhof/Münchener Straße,,de:06412:8:1:1,StrabNBörne,50.10691,8.666445,,,2,
Frankfurt (Main) Hauptbahnhof/Münchener Straße,,de:06412:8:1:4,BusNWeserMü,50.10703,8.666444,,,2,
Frankfurt (Main) Hauptbahnhof/Münchener Straße,,de:06412:8:2:2,StrabNHbf/Süd,50.106937,8.665997,,,2,
Frankfurt (Main) Hauptbahnhof/Münchener Straße,,de:06412:8:2:3,BusNHbf,50.10693,8.665997,,,2,
)";

TEST(a, b) {
  auto tt = nigiri::timetable{};
  nigiri::loader::gtfs::load_timetable({}, nigiri::source_idx_t{0},
                                       nigiri::loader::mem_dir::read(gtfs), tt);

  auto p = fs::path{"/tmp/osr_test"};
  auto ec = std::error_code{};
  fs::remove_all(p, ec);
  fs::create_directories(p, ec);

  osr::extract(true, "test/ffm_hbf.osm", "/tmp/osr_test");

  auto const w = osr::ways{p, cista::mmap::protection::READ};
  auto pl = osr::platforms{p, cista::mmap::protection::READ};
  pl.build_rtree(w);

  auto const m = icc::match(tt, pl, w);

  //  for (auto const& [l, x] : m.lp_) {
  //    fmt::println("{} ({})", tt.locations_.names_[l].view(),
  //                 tt.locations_.ids_[l].view());
  //    for (auto const& name : pl.platform_names_[x]) {
  //      fmt::println("  {}", name.view());
  //    }
  //  }
}