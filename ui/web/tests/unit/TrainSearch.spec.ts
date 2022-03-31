/* eslint-disable @typescript-eslint/no-explicit-any */
import { config, mount, VueWrapper } from "@vue/test-utils";
import { mockNextAxiosPost } from "./TestHelpers";
import flushPromises from "flush-promises";
import TrainSearch from "../../src/views/TrainSearch.vue";


describe("TrainSearch test", () => {
  let wrapper: VueWrapper;

  beforeAll(() => {
    config.global.mocks = {
      $route: {
        name: "TrainSearch"
      }
    }
  })

  beforeEach(async () => {
    mockNextAxiosPost(mockTrainGuessResponseContent);
    wrapper = mount(TrainSearch);
    await flushPromises();
  })

  it("Initial components rendered correctly", () => {
    expect(wrapper.find("div .pure-u-1 pure-u-sm-1-2 train-nr")).toBeTruthy();
    expect(wrapper.find("div .pure-u-1 pure-u-sm-12-24 to-location")).toBeTruthy();
    expect(wrapper.find("div .pure-u-1 pure-u-sm-12-24")).toBeTruthy();
  })

  it("Snapshot #1 basic structure", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it("Contet loaded correctly and snapshot #2", async done => {
    const input = wrapper.find("input");
    input.setValue("152");
    await flushPromises();
    setTimeout(() => {
      expect((wrapper.vm.$data as any).trainGuesses).toStrictEqual(mockTrainGuessResponseContent.content.trips);
      expect(wrapper.html()).toMatchSnapshot();
      // structure rendered
      expect(wrapper.find("div .trip-time")).toBeTruthy();
      expect(wrapper.find("div .trip-first-station")).toBeTruthy();
      expect(wrapper.find("div .direction")).toBeTruthy();
      // data correct
      expect((wrapper.vm.$data as any).areGuessesDisplayed).toBeTruthy();
      expect((wrapper.vm.$data as any).currentTrainInput).toBe(152);
      done();
    }, 1000);
  })
});



const mockTrainGuessResponseContent = {
  "destination": {
    "type": "Module",
    "target": ""
  },
  "content_type": "RailVizTripGuessResponse",
  "content": {
    "trips": [
      {
        "first_station": {
          "id": "swiss_8595640:0:A",
          "name": "Ziegelbrücke, Bahnhof Süd",
          "pos": {
            "lat": 47.136578,
            "lng": 9.059141
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8595640:0:A",
            "train_nr": 152,
            "time": 1646175840,
            "target_station_id": "swiss_8578498",
            "target_time": 1646176980,
            "line_id": "650"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "650",
            "name": "Bus 650",
            "provider": "Autobetrieb Weesen-Amden",
            "direction": "Amden, Vorderdorf"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:06411:24714:1:1",
          "name": "Darmstadt-Arheilgen Dreieichweg",
          "pos": {
            "lat": 49.92062,
            "lng": 8.654389
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:06411:24714:1:1",
            "train_nr": 152,
            "time": 1646191560,
            "target_station_id": "delfi_de:06432:24016:1:1",
            "target_time": 1646194980,
            "line_id": "8"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Str",
            "category_id": 1,
            "clasz": 9,
            "train_nr": 152,
            "line_id": "8",
            "name": "8",
            "provider": "HEAG Mobilo",
            "direction": "Alsbach-Hähnlein-Alsbach Am Hinkelstein"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_000005170214",
          "name": "Kostrzyn PKS",
          "pos": {
            "lat": 52.590603,
            "lng": 14.647271
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_000005170214",
            "train_nr": 152,
            "time": 1646193420,
            "target_station_id": "delfi_de:12064:900321518",
            "target_time": 1646193960,
            "line_id": "SEV"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 0,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "SEV",
            "name": "Bus SEV",
            "provider": "NEB Niederbarnimer Eisenbahn",
            "direction": "Küstrin-Kietz Bahnhof (Karl-Marx-Straße)"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8591382",
          "name": "Zürich, Sternen Oerlikon",
          "pos": {
            "lat": 47.410069,
            "lng": 8.546229
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8591382",
            "train_nr": 152,
            "time": 1646194200,
            "target_station_id": "swiss_8591065",
            "target_time": 1646195460,
            "line_id": "12"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Str",
            "category_id": 39,
            "clasz": 9,
            "train_nr": 152,
            "line_id": "12",
            "name": "Str 12",
            "provider": "Verkehrsbetriebe Zürich INFO+",
            "direction": "Stettbach, Bahnhof"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8505389",
          "name": "Tesserete, Stazione",
          "pos": {
            "lat": 46.066006,
            "lng": 8.966515
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8505389",
            "train_nr": 152,
            "time": 1646194800,
            "target_station_id": "swiss_8505913",
            "target_time": 1646197200,
            "line_id": "442"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "442",
            "name": "Bus 442",
            "provider": "PostAuto AG",
            "direction": "Lugano, Autosilo Balestra"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:04011:14184::1",
          "name": "Bremen Sebaldsbrück (Bus+Tram)",
          "pos": {
            "lat": 53.059669,
            "lng": 8.899625
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:04011:14184::1",
            "train_nr": 152,
            "time": 1646195040,
            "target_station_id": "delfi_de:04011:14243::1",
            "target_time": 1646196300,
            "line_id": "21"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 0,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "21",
            "name": "Bus 21",
            "provider": "Bremer Straßenbahn AG",
            "direction": "Universität"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:06434:16419:1:1",
          "name": "Weilrod-Hasselbach Limburger Straße",
          "pos": {
            "lat": 50.340378,
            "lng": 8.342935
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:06434:16419:1:1",
            "train_nr": 152,
            "time": 1646196420,
            "target_station_id": "delfi_de:06533:11128:2:2",
            "target_time": 1646198340,
            "line_id": "283"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 0,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "283",
            "name": "Bus 283",
            "provider": "Medenbach Traffic GmbH",
            "direction": "Bad Camberg Bahnhof"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8576178",
          "name": "Winterthur, Töss",
          "pos": {
            "lat": 47.48917,
            "lng": 8.703094
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8576178",
            "train_nr": 152,
            "time": 1646196780,
            "target_station_id": "swiss_8590980",
            "target_time": 1646198280,
            "line_id": "1"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "1",
            "name": "Bus 1",
            "provider": "Stadtbus Winterthur",
            "direction": "Winterthur, Oberwinterthur"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8578594",
          "name": "Ebnat, Wier",
          "pos": {
            "lat": 47.261978,
            "lng": 9.129435
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8578594",
            "train_nr": 152,
            "time": 1646198040,
            "target_station_id": "swiss_8506620",
            "target_time": 1646199780,
            "line_id": "770"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "770",
            "name": "Bus 770",
            "provider": "Busbetrieb Lichtensteig-Wattwil-Ebnat-Kappel",
            "direction": "Lichtensteig, Steigrüti"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8503060",
          "name": "Esslingen",
          "pos": {
            "lat": 47.287861,
            "lng": 8.709687
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8503060",
            "train_nr": 152,
            "time": 1646198220,
            "target_station_id": "swiss_8503059",
            "target_time": 1646200320,
            "line_id": "S18"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "S",
            "category_id": 27,
            "clasz": 7,
            "train_nr": 152,
            "line_id": "S18",
            "name": "S18",
            "provider": "Forchbahn",
            "direction": "Zürich Stadelhofen, Bahnhof"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_1100597",
          "name": "\u00D6tlingen (Baden), Dorfstrasse",
          "pos": {
            "lat": 47.622616,
            "lng": 7.622672
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_1100597",
            "train_nr": 152,
            "time": 1646198700,
            "target_station_id": "swiss_1104792",
            "target_time": 1646199300,
            "line_id": "12"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "12",
            "name": "Bus 12",
            "provider": "FPLAN RBG SWG",
            "direction": "Haltingen, Markgräfler Strasse"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8575651",
          "name": "Sagno, Paese",
          "pos": {
            "lat": 45.857418,
            "lng": 9.037976
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8575651",
            "train_nr": 152,
            "time": 1646199300,
            "target_station_id": "swiss_8580069",
            "target_time": 1646200500,
            "line_id": "514"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "514",
            "name": "Bus 514",
            "provider": "PostAuto AG",
            "direction": "Morbio Inferiore, Breggia"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8500258",
          "name": "Bern Münsterplattform",
          "pos": {
            "lat": 46.946575,
            "lng": 7.452523
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8500258",
            "train_nr": 152,
            "time": 1646201700,
            "target_station_id": "swiss_8500249",
            "target_time": 1646201760,
            "line_id": "ASC"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Car train",
            "category_id": 73,
            "clasz": 12,
            "train_nr": 152,
            "line_id": "ASC",
            "name": "ASC",
            "provider": "Aufzug Matte-Plattform (Bern)",
            "direction": "Bern Matte"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:03241:42",
          "name": "Hannover Hauptbahnhof/ZOB",
          "pos": {
            "lat": 52.378647,
            "lng": 9.74169
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:03241:42",
            "train_nr": 152,
            "time": 1646203680,
            "target_station_id": "delfi_de:03241:9110",
            "target_time": 1646207100,
            "line_id": "900"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 0,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "900",
            "name": "Bus 900",
            "provider": "RegioBus Hannover GmbH",
            "direction": "Burgdorf Bahnhof"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:06411:24675:1:1",
          "name": "Darmstadt Anne-Frank-Straße",
          "pos": {
            "lat": 49.847927,
            "lng": 8.620814
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:06411:24675:1:1",
            "train_nr": 152,
            "time": 1646205120,
            "target_station_id": "delfi_de:06411:8554:2:2",
            "target_time": 1646207340,
            "line_id": "H"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 0,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "H",
            "name": "Bus H",
            "provider": "HEAG mobiBus GmbH + Co. KG",
            "direction": "Darmstadt-Kranichstein Kesselhutweg"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_1400262",
          "name": "Annecy-le-Vieux, Campus",
          "pos": {
            "lat": 45.919353,
            "lng": 6.158715
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_1400262",
            "train_nr": 152,
            "time": 1646206800,
            "target_station_id": "swiss_1401153",
            "target_time": 1646209440,
            "line_id": "4"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 28,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "4",
            "name": "Bus 4",
            "provider": "SIBRA",
            "direction": "Seynod, Neigeos"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:06411:4734:48:48",
          "name": "Darmstadt Hauptbahnhof",
          "pos": {
            "lat": 49.872795,
            "lng": 8.631401
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:06411:4734:48:48",
            "train_nr": 152,
            "time": 1646207280,
            "target_station_id": "delfi_de:06411:16149:1:1",
            "target_time": 1646208300,
            "line_id": "2"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Str",
            "category_id": 1,
            "clasz": 9,
            "train_nr": 152,
            "line_id": "2",
            "name": "2",
            "provider": "HEAG Mobilo",
            "direction": "Darmstadt Böllenfalltor"
          }
        }
      },
      {
        "first_station": {
          "id": "nl_239520",
          "name": "Amsterdam, Centraal Station",
          "pos": {
            "lat": 52.380707,
            "lng": 4.899554
          }
        },
        "trip_info": {
          "id": {
            "station_id": "nl_239520",
            "train_nr": 152,
            "time": 1646207640,
            "target_station_id": "nl_1762389",
            "target_time": 1646207940,
            "line_id": "F3"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Ferry",
            "category_id": 21,
            "clasz": 11,
            "train_nr": 152,
            "line_id": "F3",
            "name": "Ferry F3",
            "provider": "GVB",
            "direction": "Buiksloterweg"
          }
        }
      },
      {
        "first_station": {
          "id": "swiss_8508453",
          "name": "Kriens (Pilatusbahn)",
          "pos": {
            "lat": 47.030418,
            "lng": 8.277876
          }
        },
        "trip_info": {
          "id": {
            "station_id": "swiss_8508453",
            "train_nr": 152,
            "time": 1646207760,
            "target_station_id": "swiss_8508455",
            "target_time": 1646209560,
            "line_id": "GB"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Telecabin",
            "category_id": 69,
            "clasz": 12,
            "train_nr": 152,
            "line_id": "GB",
            "name": "GB",
            "provider": "Kriens-Fräkmüntegg",
            "direction": "Fräkmüntegg"
          }
        }
      },
      {
        "first_station": {
          "id": "delfi_de:06411:24021:1:1",
          "name": "Darmstadt Mahatma-Gandhi-Straße",
          "pos": {
            "lat": 49.856731,
            "lng": 8.627254
          }
        },
        "trip_info": {
          "id": {
            "station_id": "delfi_de:06411:24021:1:1",
            "train_nr": 152,
            "time": 1646208780,
            "target_station_id": "delfi_de:06411:24730:1:1",
            "target_time": 1646210580,
            "line_id": "K"
          },
          "transport": {
            "range": {
              "from": 0,
              "to": 0
            },
            "category_name": "Bus",
            "category_id": 0,
            "clasz": 10,
            "train_nr": 152,
            "line_id": "K",
            "name": "Bus K",
            "provider": "HEAG mobiBus GmbH + Co. KG",
            "direction": "Darmstadt TU-Lichtwiese/Campus"
          }
        }
      }
    ]
  },
  "id": 1
}

