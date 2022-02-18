package org.example;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class HHLRBatchQueryGenerate {
  static String BASE_FLODER = "./batch_base/";

  public static void main(String[] args) throws IOException {
    String outname = "q-hhlr-hafas-pretrip-true-";
    System.out.println("Writing to " + outname);

    for (var queryFile : config) {
      var lines = queryFile.readQueryFile();
      var pkgId = 0;
      var total_id = 0;

      for (var pack : queryFile.packages) {
        System.out.println("...Writing queries for package " + pkgId);

        var currentFileName = outname + pkgId + ".txt";
        var outStream = new BufferedOutputStream(new FileOutputStream(currentFileName));

        for (var config : pack.configs) {
          System.out.println("...Adding queries for config " + config.toString());
          //expecting base files to target routing and have Default as Search Type
          for (var line : lines) {
            ++total_id;

            var out_line = line
              .replace("routing", config.targets.toString())
              .replace("Default", config.searchType.toString());

            var lastComma = out_line.lastIndexOf(',');
            out_line = out_line.substring(0, lastComma) + ", \"id\": " + total_id + "}\n";

            outStream.write(out_line.getBytes(StandardCharsets.UTF_8));
          }
        }

        ++pkgId;
        outStream.close();
      }
    }
  }

  static QueryFile[] config = new QueryFile[]{
//    new QueryFile(Dataset.swiss, QueryType.OntripStationStart, false, new Package[]{
//      new Package(
//        new Config[]{
//          new Config(Targets.raptor_gpu, SearchType.Default),
//          new Config(Targets.raptor_gpu, SearchType.MaxOccupancy),
//          new Config(Targets.raptor_gpu, SearchType.MaxTransferClass),
//          new Config(Targets.raptor_gpu, SearchType.Tso96),
//          new Config(Targets.raptor_gpu, SearchType.Tso90),
//          new Config(Targets.raptor_gpu, SearchType.Tso80),
//          new Config(Targets.raptor_gpu, SearchType.Tso72),
//        }),
//
//      new Package(new Config[]{
//          new Config(Targets.raptor_gpu, SearchType.Tso64),
//          new Config(Targets.raptor_gpu, SearchType.Tso45),
//          new Config(Targets.raptor_gpu, SearchType.Tso36),
//          new Config(Targets.raptor_gpu, SearchType.Tso32),
//          new Config(Targets.raptor_gpu, SearchType.Tso20),
//          new Config(Targets.raptor_gpu, SearchType.Tso16),
//          new Config(Targets.raptor_gpu, SearchType.Tso10),
//          new Config(Targets.raptor_gpu, SearchType.Tso06),
//      }),
//
//      new Package(new Config[] {
//          new Config(Targets.raptor_gpu, SearchType.Tso60),
//          new Config(Targets.raptor_gpu, SearchType.Tso48),
//          new Config(Targets.raptor_gpu, SearchType.Tso40),
//          new Config(Targets.raptor_gpu, SearchType.Tso30),
//          new Config(Targets.raptor_gpu, SearchType.Tso24),
//          new Config(Targets.raptor_gpu, SearchType.Tso18),
//          new Config(Targets.raptor_gpu, SearchType.Tso12),
//          new Config(Targets.raptor_gpu, SearchType.Tso08),
//      }),

//      //duration 7h
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Default)}),
//
//      //duration 11h
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.MaxOccupancy)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso06)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso08)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso12)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso16)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso18)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso20)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso24)}),
//
//      //duration 20h
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso30)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso32)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso36)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso40)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso45)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso48)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso60)}),
//
//      //duration 24h
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso64)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso72)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso80)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso90)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.Tso96)}),


    //HAFAS CPU
//      new Package(new Config[]{new Config(Targets.routing, SearchType.Default),}),
//      new Package(new Config[]{new Config(Targets.routing, SearchType.MaxOccupancy),}),
//      new Package(new Config[]{new Config(Targets.routing, SearchType.TimeSlottedOccupancy),}),
//      new Package(new Config[]{new Config(Targets.routing, SearchType.MaxTransferClass)}),
//      new Package(new Config[]{new Config(Targets.raptor_cpu, SearchType.MaxTransferClass)})

    //SWISS GPU continuity runs
//      new Package(
//        new Config[] {
//          new Config(Targets.raptor_gpu, SearchType.Tso20),
//          new Config(Targets.raptor_gpu, SearchType.Tso24),
//          new Config(Targets.raptor_gpu, SearchType.Tso30),
//          new Config(Targets.raptor_gpu, SearchType.Tso32),
//          new Config(Targets.raptor_gpu, SearchType.Tso36),
//          new Config(Targets.raptor_gpu, SearchType.Tso40),
//          new Config(Targets.raptor_gpu, SearchType.Tso45),
//          new Config(Targets.raptor_gpu, SearchType.Tso48),
//          new Config(Targets.raptor_gpu, SearchType.Tso60),
//          new Config(Targets.raptor_gpu, SearchType.Tso64),
//          new Config(Targets.raptor_gpu, SearchType.Tso72),
//          new Config(Targets.raptor_gpu, SearchType.Tso80),
//          new Config(Targets.raptor_gpu, SearchType.Tso90),
//          new Config(Targets.raptor_gpu, SearchType.Tso96),
//        }
//      )

//    });

    new QueryFile(Dataset.hafas, QueryType.PretripStart, true, new Package[]{
      new Package(
        new Config[]{
          new Config(Targets.raptor_cpu, SearchType.Default),
          new Config(Targets.raptor_cpu, SearchType.MaxOccupancy),
          new Config(Targets.raptor_cpu, SearchType.MaxTransferClass),
          new Config(Targets.raptor_cpu, SearchType.Tso20),
          new Config(Targets.raptor_cpu, SearchType.Tso24),
          new Config(Targets.raptor_cpu, SearchType.Tso30),
          new Config(Targets.raptor_cpu, SearchType.Tso32),
          new Config(Targets.raptor_cpu, SearchType.Tso36),
        }),
      new Package(
        new Config[]{
          new Config(Targets.raptor_cpu, SearchType.Tso96),
          new Config(Targets.raptor_cpu, SearchType.Tso60),
          new Config(Targets.raptor_cpu, SearchType.Tso40),
        }
      ),
      new Package(
        new Config[]{
          new Config(Targets.raptor_cpu, SearchType.Tso90),
          new Config(Targets.raptor_cpu, SearchType.Tso64),
          new Config(Targets.raptor_cpu, SearchType.Tso45),
        }
      ),
      new Package(
        new Config[]{
          new Config(Targets.raptor_cpu, SearchType.Tso80),
          new Config(Targets.raptor_cpu, SearchType.Tso72),
          new Config(Targets.raptor_cpu, SearchType.Tso48),
        }
      ),

      new Package(
        new Config[]{
          new Config(Targets.raptor_gpu, SearchType.Default),
          new Config(Targets.raptor_gpu, SearchType.MaxOccupancy),
          new Config(Targets.raptor_gpu, SearchType.MaxTransferClass),
          new Config(Targets.raptor_gpu, SearchType.Tso20),
          new Config(Targets.raptor_gpu, SearchType.Tso24),
          new Config(Targets.raptor_gpu, SearchType.Tso30),
          new Config(Targets.raptor_gpu, SearchType.Tso32),
          new Config(Targets.raptor_gpu, SearchType.Tso36),
        }),
      new Package(
        new Config[]{
          new Config(Targets.raptor_gpu, SearchType.Tso96),
          new Config(Targets.raptor_gpu, SearchType.Tso60),
          new Config(Targets.raptor_gpu, SearchType.Tso40),
        }
      ),
      new Package(
        new Config[]{
          new Config(Targets.raptor_gpu, SearchType.Tso90),
          new Config(Targets.raptor_gpu, SearchType.Tso64),
          new Config(Targets.raptor_gpu, SearchType.Tso45),
        }
      ),
      new Package(
        new Config[]{
          new Config(Targets.raptor_gpu, SearchType.Tso80),
          new Config(Targets.raptor_gpu, SearchType.Tso72),
          new Config(Targets.raptor_gpu, SearchType.Tso48),
        }
      ),

      new Package(
        new Config[]{
          new Config(Targets.routing, SearchType.Default),
          new Config(Targets.routing, SearchType.MaxOccupancy),
          new Config(Targets.routing, SearchType.MaxTransferClass),
          new Config(Targets.routing, SearchType.TimeSlottedOccupancy),
        }
      )
    })
  };


  static class QueryFile {

    public QueryFile(Dataset dataset, QueryType queryType, boolean largeStation, Package[] packages) {
      this.dataset = dataset;
      this.queryType = queryType;
      this.largeStation = largeStation;
      this.packages = packages;
    }

    Dataset dataset;
    QueryType queryType;
    boolean largeStation;

    Package[] packages;

    List<String> readQueryFile() {
      var fileName = "q-" + dataset.toString().toLowerCase() + "-" + queryType.toString().toLowerCase() + "-" + largeStation + ".txt";
//      var fileName = "queries-set-mix-raptor_gpu-3.txt";
      var path = BASE_FLODER + fileName;

      System.out.print("Reading file '" + path + "' ...");
      try {
        var input = Files.readAllLines(Path.of(path));
        System.out.println("Ok");
        return input;
      } catch (IOException ex) {
        System.out.println("Error!");
        ex.printStackTrace();
      }

      System.exit(-1);
      return null;
    }
  }

  static class Package {
    public Package(Config[] configs) {
      this.configs = configs;
    }

    Config[] configs;
  }

  static class Config {
    public Config(Targets targets, SearchType searchType) {
      this.targets = targets;
      this.searchType = searchType;
    }

    Targets targets;
    SearchType searchType;

    @Override
    public String toString() {
      return "Config{" +
        "targets=" + targets +
        ", searchType=" + searchType +
        '}';
    }
  }
}

enum SearchType {
  Default,
  MaxOccupancy,
  TimeSlottedOccupancy,

  MaxTransferClass,

  Tso96,
  Tso90,
  Tso80,
  Tso72,
  Tso64,
  Tso60,
  Tso48,
  Tso45,
  Tso40,
  Tso36,
  Tso32,
  Tso30,
  Tso24,
  Tso20,
  Tso18,
  Tso16,
  Tso12,
  Tso10,
  Tso08,
  Tso06,
}

enum QueryType {
  OntripStationStart,
  PretripStart
}

enum Dataset {
  swiss,
  hafas,
  delfi,
  vrn;
}

enum Targets {
  routing,
  raptor_cpu,
  raptor_gpu;
}
