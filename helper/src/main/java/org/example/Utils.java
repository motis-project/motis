package org.example;

public class Utils {

  static final long SCHEDULE_BEGIN = 1629504000;
  public static long unix_to_motis_time(long unix_time) {
    if (unix_time < SCHEDULE_BEGIN) {
      throw new IllegalStateException("Time!");
    }
    return (unix_time - SCHEDULE_BEGIN) / 60;
  }

  public static void main(String[] args) {
    System.out.println(unix_to_motis_time(1630006500));
    System.out.println(unix_to_motis_time(1630006920));
    System.out.println(unix_to_motis_time(1630015020));
    System.out.println();
    System.out.println(unix_to_motis_time(1630015260));
    System.out.println(unix_to_motis_time(1630015740));
    System.out.println(unix_to_motis_time(1630015980));
    System.out.println();
    System.out.println(unix_to_motis_time(1630016220));
    System.out.println(unix_to_motis_time(1630017840));
  }

}
