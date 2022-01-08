package org.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class UpdateAnalysisCache {

  final static String MOTIS_EXEC = "C:\\Users\\Leon\\git\\motis\\cmake-build-relwithdebinfo\\motis";

  final static String ANALYZE_DIR = "C:\\Users\\Leon\\git\\motis\\Auswertung\\Swiss\\SplitResults\\";

  static List<String> analyze(Path file) throws IOException, InterruptedException {
    var p = new ProcessBuilder(MOTIS_EXEC, "analyze", file.toString())
      .redirectError(ProcessBuilder.Redirect.INHERIT)
      .redirectOutput(ProcessBuilder.Redirect.PIPE)
      .start();

    var inStream = p.getInputStream();
    var pStdOut = new String(inStream.readAllBytes());
    return Arrays.stream(pStdOut.split("\n\r")).toList();
  }

  

  public static void main(String[] args) throws IOException, InterruptedException {
    var inFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> !p.getFileName().toString().endsWith(".anlz"))
      .toList();

    var cacheFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> p.getFileName().toString().endsWith(".anlz"))
      .toList();

    for(var inFile : inFiles) {
      System.out.println("Processing File '" + inFile + "'");
      //try to find existing cache file
      var found = false;
      for(var cache : cacheFiles) {
        if(cache.getFileName().toString().startsWith(inFile.getFileName().toString())) {
          found = true;
          System.out.println("  ... found cache file.");
          break;
        }
      }

      if(found){
        System.out.println("  ... skipping.");
        continue;
      }

      System.out.println("  ... analyzing");
      var analysis = analyze(inFile);
      System.out.println("  ... analysis done.");

      var outFileName = inFile.getFileName().toString() + ".anlz";
      var outDir = inFile.getParent() + "/" + outFileName;
      System.out.println("  ... determined cache file name '" + outDir + "'");

      Files.write(Path.of(outDir), analysis);
      System.out.println("  ... wrote cache file.");
    }
  }
}
