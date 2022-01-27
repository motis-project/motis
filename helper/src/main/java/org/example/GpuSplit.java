package org.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class GpuSplit {
  public static void main(String[] args) throws IOException {
    var path = Path.of("C:\\Users\\Leon\\git\\motis\\Auswertung\\Hafas\\HHLRResults\\r-hhlr-v100-hafas-ontrip-false-raptor-gpu.txt");
    var outPath = Path.of("C:\\Users\\Leon\\git\\motis\\Auswertung\\Hafas\\HHLRResults\\r-hhlr-v100-hafas-ontrip-false-raptor-gpu-0.txt");
    var br = Files.newBufferedReader(path);
    var out = Files.newBufferedWriter(outPath);

    for (int i = 0; i < 70000; i++) {
      var line = br.readLine();
      out.write(line);
      out.write("\n");

      if (i%1000 == 0)
        System.out.println("..." + i);
    }

    br.close();
    out.close();
  }
}
