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
    String outname = "q-hhlr-batch.txt";
    System.out.println("Writing to " + outname);

    var outStream = new BufferedOutputStream(new FileOutputStream(outname));

    for(var queryFile : config) {
      var total_id = 0;
      var lines = queryFile.readQueryFile();
      for(var config : queryFile.configs) {
        System.out.print("Writing queries for config " + config.toString() + "...");
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

        System.out.println("Ok");
      }
    }

    outStream.close();
  }

  static QueryFile[] config = new QueryFile[]{
    new QueryFile(Dataset.swiss, QueryType.OntripStationStart, false, new Config[]{
      //Default
      new Config(Targets.raptor_cpu, SearchType.Default),
      new Config(Targets.raptor_gpu, SearchType.Default),

      //Max Occupancy
      new Config(Targets.raptor_cpu, SearchType.MaxOccupancy),
      new Config(Targets.raptor_gpu, SearchType.MaxOccupancy),
      new Config(Targets.raptor_gpu, SearchType.MaxOccupancyShfl),

      //Time Slotted Occupancy
      new Config(Targets.raptor_cpu, SearchType.Tso96),
      new Config(Targets.raptor_cpu, SearchType.Tso90),
      new Config(Targets.raptor_cpu, SearchType.Tso80),
      new Config(Targets.raptor_cpu, SearchType.Tso72),
      new Config(Targets.raptor_cpu, SearchType.Tso64),
      new Config(Targets.raptor_cpu, SearchType.Tso60),
      new Config(Targets.raptor_cpu, SearchType.Tso48),
      new Config(Targets.raptor_cpu, SearchType.Tso45),
      new Config(Targets.raptor_cpu, SearchType.Tso40),
      new Config(Targets.raptor_cpu, SearchType.Tso36),
      new Config(Targets.raptor_cpu, SearchType.Tso32),
      new Config(Targets.raptor_cpu, SearchType.Tso30),
      new Config(Targets.raptor_cpu, SearchType.Tso24),
      new Config(Targets.raptor_cpu, SearchType.Tso20),
      new Config(Targets.raptor_cpu, SearchType.Tso18),
      new Config(Targets.raptor_cpu, SearchType.Tso16),
      new Config(Targets.raptor_cpu, SearchType.Tso12),
      new Config(Targets.raptor_cpu, SearchType.Tso10),
      new Config(Targets.raptor_cpu, SearchType.Tso08),
      new Config(Targets.raptor_cpu, SearchType.Tso06),

      new Config(Targets.raptor_gpu, SearchType.Tso96),
      new Config(Targets.raptor_gpu, SearchType.Tso90),
      new Config(Targets.raptor_gpu, SearchType.Tso80),
      new Config(Targets.raptor_gpu, SearchType.Tso72),
      new Config(Targets.raptor_gpu, SearchType.Tso64),
      new Config(Targets.raptor_gpu, SearchType.Tso60),
      new Config(Targets.raptor_gpu, SearchType.Tso48),
      new Config(Targets.raptor_gpu, SearchType.Tso45),
      new Config(Targets.raptor_gpu, SearchType.Tso40),
      new Config(Targets.raptor_gpu, SearchType.Tso36),
      new Config(Targets.raptor_gpu, SearchType.Tso32),
      new Config(Targets.raptor_gpu, SearchType.Tso30),
      new Config(Targets.raptor_gpu, SearchType.Tso24),
      new Config(Targets.raptor_gpu, SearchType.Tso20),
      new Config(Targets.raptor_gpu, SearchType.Tso18),
      new Config(Targets.raptor_gpu, SearchType.Tso16),
      new Config(Targets.raptor_gpu, SearchType.Tso12),
      new Config(Targets.raptor_gpu, SearchType.Tso10),
      new Config(Targets.raptor_gpu, SearchType.Tso08),
      new Config(Targets.raptor_gpu, SearchType.Tso06),

      new Config(Targets.raptor_gpu, SearchType.Tso96Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso90Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso80Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso72Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso64Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso60Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso48Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso45Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso40Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso36Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso32Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso30Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso24Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso20Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso18Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso16Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso12Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso10Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso08Shfl),
      new Config(Targets.raptor_gpu, SearchType.Tso06Shfl),
    })

  };

  static class QueryFile {

    public QueryFile(Dataset dataset, QueryType queryType, boolean largeStation, Config[] configs) {
      this.dataset = dataset;
      this.queryType = queryType;
      this.largeStation = largeStation;
      this.configs = configs;
    }

    Dataset dataset;
    QueryType queryType;
    boolean largeStation;

    Config[] configs;

    List<String> readQueryFile() {
      var fileName = "q-" + dataset.toString().toLowerCase() + "-" + queryType.toString().toLowerCase() + "-" + largeStation + ".txt";
      var path = BASE_FLODER + fileName;

      System.out.print("Reading file '" + path + "' ...");
      try {
        var input = Files.readAllLines(Path.of(path));
        System.out.println("Ok");
        return input;
      }catch(IOException ex) {
        System.out.println("Error!");
        ex.printStackTrace();
      }

      System.exit(-1);
      return null;
    }
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
  MaxOccupancyShfl,

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

  Tso96Shfl,
  Tso90Shfl,
  Tso80Shfl,
  Tso72Shfl,
  Tso64Shfl,
  Tso60Shfl,
  Tso48Shfl,
  Tso45Shfl,
  Tso40Shfl,
  Tso36Shfl,
  Tso32Shfl,
  Tso30Shfl,
  Tso24Shfl,
  Tso20Shfl,
  Tso18Shfl,
  Tso16Shfl,
  Tso12Shfl,
  Tso10Shfl,
  Tso08Shfl,
  Tso06Shfl
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
